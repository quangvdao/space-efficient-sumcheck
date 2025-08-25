use crate::{
	messages::VerifierMessages,
	multilinear_product::TimeProductProver,
	order_strategy::SignificantBitOrder,
	streams::{Stream, StreamIterator},
	interpolation::{field_mul_small::FieldMulSmall, LagrangePolynomial},
};
use crate::interpolation::multivariate::{compute_strides, multi_product_eval_paper};
use crate::hypercube::Hypercube;
use ark_ff::Field;
use ark_std::vec::Vec;
use std::collections::BTreeSet;

// Evaluate a degree-≤D polynomial represented on U_D = [1,2,...,D, ∞] at an arbitrary point r.
// `line` layout: [p(1), p(2), ..., p(D), s_inf].
#[inline]
fn eval_from_u_d_at_point<F: Field>(line: &[F], r: F) -> F {
    let d = line.len().saturating_sub(1);
    debug_assert!(d >= 1);
    let s_inf = line[d];
    let finite_values = &line[..d];
    let mut finite_nodes: Vec<F> = Vec::with_capacity(d);
    for i in 0..d { finite_nodes.push(F::from((i as u32) + 1)); }
    LagrangePolynomial::<F, SignificantBitOrder>::evaluate_from_infty_and_points(
        r,
        s_inf,
        &finite_nodes,
        finite_values,
    )
}

// Compute f(0)+f(1) given a univariate degree-≤D polynomial specified by values on U_D = [1..D, ∞].
// `line` layout: [f(1), f(2), ..., f(D), c_d] where c_d is the leading coefficient (value at ∞).
#[inline]
fn sum_bool_from_u_d_line<F: Field>(line: &[F]) -> F {
    // Evaluate f at 0 and 1 from values on U_D and add
    let f0 = eval_from_u_d_at_point::<F>(line, F::ZERO);
    let f1 = eval_from_u_d_at_point::<F>(line, F::ONE);
    f0 + f1
}

/// Implements the evaluation-basis sum-check prover described in `sections/5_small_value_streaming.tex`
/// (Figure `fig:small-value-streaming`, `EvalProduct`).
///
/// This prover uses a **round-batching** or **windowing** strategy. Instead of streaming the
/// input polynomials for each round, it performs a single, expensive pass to compute a "window
/// polynomial" `q_t` that is sufficient to answer challenges for a whole window of `ω_t`
/// rounds. This material is stored in an evaluation grid (`reduced_grid`).
///
/// The protocol proceeds in phases, managed by a `window_schedule`:
/// 1.  **Window Computation**: At the start of a window (pass), `build_window_grid` is called.
///     It computes `q_t` by summing out non-window variables and stores its evaluations on U_D^ω.
/// 2.  **Round Computation**: For each round within the window, the prover uses the in-memory grid
///     to compute the round polynomial (`compute_round`) and bind challenges (`collapse_axis_at_point`),
///     which is equivalent to running `LinearTime_SC` on `q_t`.
/// 3.  **Final Phase**: For the last few rounds, it can switch to a standard `TimeProductProver`
///     to avoid the overhead of windowing on small inputs.
pub struct StreamingEvalProductProver<F: Field, S: Stream<F>, const D: usize> {
	pub claim: F,
	pub current_round: usize,
	pub streams: [S; D],
	pub stream_iterators: [StreamIterator<F, S, SignificantBitOrder>; D],
	pub num_stages: usize,
	pub num_variables: usize,
	pub last_round_phase1: usize,
	pub verifier_messages: VerifierMessages<F>,
	pub verifier_messages_round_comp: VerifierMessages<F>,
	pub stage_size: usize,
	pub prev_table_round_num: usize,
	pub prev_table_size: usize,
	pub state_comp_set: BTreeSet<usize>,
	pub switched_to_vsbw: bool,
	pub vsbw_prover: TimeProductProver<F, S, D>,
	// Window runtime state  
	pub current_window_size: usize,
	pub window_offset: usize,
	pub reduced_grid: Option<Vec<F>>, // evaluation grid for remaining axes (A^{ω-Δ})
	pub reduced_shape: Vec<usize>,    // shape per axis (each D+1)
}

impl<F: Field + FieldMulSmall, S: Stream<F>, const D: usize> StreamingEvalProductProver<F, S, D> {
	pub fn is_initial_round(&self) -> bool {
		self.current_round == 0
	}

	pub fn total_rounds(&self) -> usize {
		self.num_variables
	}

	pub fn init_round_vars(&mut self) {
		let n = self.num_variables;
		let j = self.current_round + 1;

		if let Some(&prev_round) = self.state_comp_set.range(..=j).next_back() {
			self.prev_table_round_num = prev_round;
			if let Some(&next_round) = self.state_comp_set.range((j + 1)..).next() {
				self.prev_table_size = next_round - prev_round;
			} else {
				self.prev_table_size = n + 1 - prev_round;
			}
		} else {
			self.prev_table_round_num = 0;
			self.prev_table_size = 0;
		}
	}

	/// Binds the first variable of the window polynomial `q_t` to the verifier's challenge `r`.
	/// This corresponds to the update step of `LinearTime_SC` applied to the window polynomial.
	///
	/// It operates on the `reduced_grid` by taking each (v-1)-dimensional slice and reducing
	/// it to a (v-1)-dimensional point. This is done by gathering the evaluations along the
	/// first axis into a line and finding the value of the corresponding univariate polynomial
	/// at `r` using `eval_from_u_d_at_point`. The result is a new, smaller grid with one
	/// fewer dimension, ready for the next round's computation.
	fn collapse_axis_at_point(&mut self, r: F) {
		if self.reduced_grid.is_none() { return; }
		let grid = self.reduced_grid.as_ref().unwrap();
		let shape = &self.reduced_shape;
		if shape.is_empty() { return; }
		let strides = compute_strides(shape);
		let axis_len = shape[0];
		// New shape excludes axis 0
		let mut out_shape: Vec<usize> = Vec::with_capacity(shape.len().saturating_sub(1));
		for i in 1..shape.len() { out_shape.push(shape[i]); }
		let out_elems = out_shape.iter().copied().fold(1usize, |acc, s| acc * s);
		let mut out = vec![F::ZERO; out_elems];
		for out_idx in 0..out_elems {
			// Map index to coordinates for axes 1..v-1
			let mut rem = out_idx;
			let mut src_base = 0usize;
			for i in 1..shape.len() {
				let dim = out_shape[i - 1];
				let coord = rem % dim;
				rem /= dim;
				src_base += coord * strides[i];
			}
			// Gather line along axis 0
			let mut line: Vec<F> = Vec::with_capacity(axis_len);
			for t in 0..axis_len { line.push(grid[src_base + t * strides[0]]); }
			out[out_idx] = eval_from_u_d_at_point::<F>(&line, r);
		}
		self.reduced_grid = Some(out);
		self.reduced_shape = out_shape;
		self.window_offset += 1;
	}

	#[inline]
	fn sum_over_remaining_axes_at_axis0_index(&self, idx0: usize) -> F {
		if self.reduced_grid.is_none() { return F::ZERO; }
		let grid = self.reduced_grid.as_ref().unwrap();
		let shape = &self.reduced_shape;
		if shape.is_empty() { return F::ZERO; }
		let strides = compute_strides(shape);
		let total_rest = shape[1..].iter().copied().fold(1usize, |acc, s| acc * s);
		let mut acc = F::ZERO;
		for rest_idx in 0..total_rest {
			// map rest_idx to offset over axes 1..v-1
			let mut rem = rest_idx;
			let mut offset = idx0 * strides[0];
			for i in 1..shape.len() {
				let dim = shape[i];
				let coord = rem % dim;
				rem /= dim;
				offset += coord * strides[i];
			}
			acc += grid[offset];
		}
		acc
	}

	/// Corresponds to step 1(a) in Figure `fig:small-value-streaming`.
	///
	/// This function computes the evaluations of the window polynomial `q_t` on the grid U_D^ω.
	/// The paper defines `q_t` as:
	///   `q_t(X_1..X_ω) = Σ_{x'} Π_{k=1..d} p_k(r_{<S_t}, X_1..X_ω, x')`
	///
	/// This implementation breaks the computation into several parts:
	/// 1.  **Variable Splitting**: It partitions the `num_variables` into three sets:
	///     - `x_num_vars`: Variables already bound to challenges `r_{<S_t}`.
	///     - `omega`: The `ω` variables of the current window (X_1..X_ω).
	///     - `b_num_vars`: The trailing variables `x'` that must be summed out.
	/// 2.  **Lagrange Weights**: It computes `weights_by_x`, the Lagrange coefficients for the
	///     bound variables, to efficiently evaluate `p_k(r_{<S_t}, ...)` from the initial streams.
	/// 3.  **Summation and Product**: It iterates through all assignments to the trailing `b_num_vars` (`x'`).
	///     In each iteration, it:
	///     a. Computes the evaluations of each `p_k(r_{<S_t}, B', b)` for all `B' ∈ {0,1}^ω`.
	///        This yields `ml_inputs_01`, `d` multilinear polynomials in `ω` variables (each on {0,1}^ω).
	///     b. Calls `multivariate_product_evaluations` (the `MultiProductEval` routine) to compute
	///        the product `Π p_k` on the extended grid U_D^ω.
	///     c. Accumulates this result into `self.reduced_grid`, performing the outer `Σ_{x'}`.
	///
	/// A fast path for `omega = 1` is included as a performance optimization, as the general
	/// multivariate machinery is not needed for a univariate window polynomial.
	fn build_window_grid(&mut self, omega: usize)
	where
		F: FieldMulSmall,
	{
		// Derive dimensions
		let x_num_vars = self.current_round; // window starts at current round (1-indexed)
		let b_num_vars = self.num_variables - x_num_vars - omega;
		assert!(b_num_vars as isize >= 0, "invalid window dims");
		// Precompute Lagrange weights for prefix x over all 2^{x} assignments, keyed by lex index
		let mut weights_by_x: Vec<F> = vec![F::ONE; 1usize << x_num_vars];
		if x_num_vars > 0 {
			let mut sequential_lag_poly = LagrangePolynomial::<F, SignificantBitOrder>::new(&self.verifier_messages);
			for (x_index_lex, _) in Hypercube::<SignificantBitOrder>::new(x_num_vars) {
				weights_by_x[x_index_lex] = sequential_lag_poly.next().unwrap();
			}
		}

		// Build inverse permutation from lex index -> position in SignificantBitOrder sequence
		let total_points = 1usize << self.num_variables;
		let mut pos_of_lex: Vec<usize> = vec![0usize; total_points];
		{
			let mut pos: usize = 0;
			for (lex_idx, _) in Hypercube::<SignificantBitOrder>::new(self.num_variables) {
				pos_of_lex[lex_idx] = pos;
				pos += 1;
			}
		}

		// For each trailing b assignment, build per-poly window slices on {0,1}^{omega} and accumulate product on U_D^{omega}
		let num_b = 1usize << b_num_vars;
		for b_index in 0..num_b {
			let mut ml_inputs_01: Vec<Vec<F>> = Vec::with_capacity(D);
			for j in 0..D {
				let mut window_vals_01: Vec<F> = vec![F::ZERO; 1usize << omega];
				for b_prime_index in 0..(1usize << omega) {
					let mut acc = F::ZERO;
					if x_num_vars == 0 {
						// lex_idx encodes [x | b' | b] with x empty
						let lex_idx = (b_prime_index << b_num_vars) | b_index;
						let pos = pos_of_lex[lex_idx];
						let v = self.streams[j].evaluation(pos);
						acc += v;
					} else {
						for x_index in 0..(1usize << x_num_vars) {
							// Compose lex index: x in the most significant bits, then b', then b
							let lex_idx = (x_index << (omega + b_num_vars)) | (b_prime_index << b_num_vars) | b_index;
							let pos = pos_of_lex[lex_idx];
							let v = self.streams[j].evaluation(pos);
							acc += weights_by_x[x_index] * v;
						}
					}
					window_vals_01[b_prime_index] = acc;
				}
				ml_inputs_01.push(window_vals_01);
			}
			let prod = multi_product_eval_paper::<F>(omega, &ml_inputs_01, D);
			let grid = self.reduced_grid.as_mut().expect("grid should be allocated");
			for i in 0..grid.len() { grid[i] += prod[i]; }
		}
	}

	/// Computes the prover's message for the current round from the precomputed window grid.
	/// This corresponds to one step of running `LinearTime_SC` on the window polynomial `q_t`.
	///
	/// For a round corresponding to window variable `X_j`, the prover must compute:
	///   `s_j(X_j) = Σ_{x'' ∈ {0,1}^{ω-j}} q_t(r_1, ..., r_{j-1}, X_j, x'')`
	///
	/// This is achieved by taking the current `reduced_grid` (which has `ω-(j-1)` dimensions)
	/// and summing out all dimensions except for the first one (which corresponds to `X_j`).
	/// The summation over the Boolean hypercube is performed by `sum_bool_from_u_d_line`,
	/// which evaluates a univariate polynomial at 0 and 1 and adds the results. This is
	/// repeated for each remaining axis until only a single line of evaluations for `s_j`
	/// remains.
	pub fn compute_round(&mut self) -> Vec<F> {
		if self.switched_to_vsbw {
			return self.vsbw_prover.vsbw_evaluate();
		}
		if D == 1 {
			let mut out = vec![F::ZERO; 1];
			out[0] = self.sum_over_remaining_axes_at_axis0_index(0); // node 1 -> idx 0
			return out;
		}
		let mut out = vec![F::ZERO; D];
		if self.reduced_grid.is_none() { return out; }
		let grid = self.reduced_grid.as_ref().unwrap().clone();
		let mut shape = self.reduced_shape.clone();
		// Reduce axes 1.. to a single axis by summing over Booleans via mu-weights
		let mut cur = grid;
		// collapse axes one by one using Boolean-sum from U_D lines
		// Iteratively collapse axis 1.. until only axis 0 remains
		while shape.len() > 1 {
			let axis = 1usize; // always collapse the next axis after 0
			let strides = compute_strides(&shape);
			let src_axis_len = shape[axis];
			debug_assert_eq!(src_axis_len, D + 1);
			let mut next_shape = shape.clone();
			next_shape.remove(axis);
			let total_next = next_shape.iter().copied().fold(1usize, |acc, s| acc * s);
			let mut next = vec![F::ZERO; total_next];
			for out_idx in 0..total_next {
				// map out_idx to coords over remaining axes (including axis 0)
				let mut rem = out_idx;
				let mut src_base = 0usize;
				for i in 0..shape.len() {
					if i == axis { continue; }
					let dim = if i < axis { next_shape[i] } else { next_shape[i - 1] };
					let coord = rem % dim;
					rem /= dim;
					src_base += coord * strides[i];
				}
				// gather line along the collapsed axis and apply mu-sum
				let mut line: Vec<F> = Vec::with_capacity(src_axis_len);
				for t in 0..src_axis_len { line.push(cur[src_base + t * strides[axis]]); }
				next[out_idx] = sum_bool_from_u_d_line::<F>(&line);
			}
			cur = next;
			shape = next_shape;
		}
		// Now cur is a single line along axis 0 of length D+1 in U_D order [∞, 0, 1, ..., D-1]
		// Extract in sumcheck format: [f(1), f(2), ..., f(D-1), f(∞)]
		for z in 1..D { out[z - 1] = cur[z]; }  // out[0] = cur[1] = f(1), out[1] = cur[2] = f(2), etc.
		out[D - 1] = cur[0];  // out[D-1] = cur[0] = f(∞)
		println!("DEBUG Streaming: compute_round out={:?}", out);
		out
	}

	/// The main state machine for the prover, called once per round.
	///
	/// This function implements the top-level loop ("For t=1..T") from Figure `fig:small-value-streaming`.
	/// It manages the windowing lifecycle:
	///
	/// 1.  **Switch to Final Prover**: First, it checks if the protocol has reached the final rounds.
	///     If so, it switches to the more efficient `vsbw_prover` (`TimeProductProver`), which
	///     implements the classic `LinearTime_SC`.
	///
	/// 2.  **Start of a New Window**: If no `reduced_grid` exists, it means a new pass/window must
	///     begin. It allocates a zero-initialized grid of shape (D+1)^ω and calls `build_window_grid`
	///     to populate it. This is the most expensive part of the protocol.
	///
	/// 3.  **Inside an Active Window**: If a `reduced_grid` exists, it means we are in the middle of a
	///     window. The function calls `collapse_axis_at_point` to bind the first axis of the grid
	///     to the verifier's challenge from the previous round. This corresponds to the update step
	///     of `LinearTime_SC` on the window polynomial `q_t`.
	///
	/// 4.  **End of a Window**: If `collapse_axis_at_point` collapses the last dimension of the grid,
	///     the grid is cleared. This signals that the window is complete, and a new one will be
	///     built on the next call.
	pub fn compute_state(&mut self) {
		let j = self.current_round + 1;
		let p = self.state_comp_set.contains(&j);
		let is_largest = self.state_comp_set.range((j + 1)..).next().is_none();
		
		if p && !is_largest {
			// Start of a new window pass: compute window size and build grid
			let j_prime = j; // Current round is the start of this window
			
			// Compute window size as distance to next pass start (or end of protocol)
			let omega = if let Some(&next_round) = self.state_comp_set.range((j + 1)..).next() {
				next_round - j
			} else {
				self.num_variables + 1 - j
			};
			
			println!(
				"DEBUG EvalProduct: Starting window pass at round {}, j_prime={}, omega={}",
				j, j_prime, omega
			);
			
			// Allocate and populate the window grid
			self.current_window_size = omega;
			self.reduced_shape = core::iter::repeat(D + 1).take(omega).collect();
			let total = self.reduced_shape.iter().copied().fold(1usize, |acc, s| acc * s);
			self.reduced_grid = Some(vec![F::ZERO; total]);
			self.window_offset = 0;
			self.build_window_grid(omega);
		} else if p && is_largest {
			// Switch to VSBW for final rounds
			let num_variables_new = self.num_variables - j + 1;
			self.switched_to_vsbw = true;
			
			println!(
				"DEBUG EvalProduct: Switching to VSBW at round {}, remaining vars={}",
				j, num_variables_new
			);
			
			self.stream_iterators.iter_mut().for_each(|it| it.reset());
			
			// Initialize VSBW prover to read from streams (not pre-computed evaluations)
			for t in 0..D { 
				self.vsbw_prover.evaluations[t] = None; 
			}
			self.vsbw_prover.current_round = 0;
			self.vsbw_prover.num_variables = num_variables_new;
			self.vsbw_prover.streams = Some(self.streams.clone());
		} else if self.switched_to_vsbw {
			// Continue with VSBW
			let verifier_message = self.verifier_messages.messages[self.current_round - 1];
			self.vsbw_prover.vsbw_reduce_evaluations(verifier_message);
		} else {
			// Within a window: collapse axis if we have a challenge from previous round
			if self.current_round > 0 && self.reduced_grid.is_some() {
				if self.reduced_shape.len() > 1 {
					let r = self.verifier_messages.messages[self.current_round - 1];
					self.collapse_axis_at_point(r);
					
					// Check if window is finished
					if self.reduced_shape.is_empty() {
						self.reduced_grid = None;
						self.current_window_size = 0;
						self.window_offset = 0;
					}
				} else if self.reduced_shape.len() <= 1 {
					// Window with single axis is finished
					self.reduced_grid = None;
					self.current_window_size = 0;
					self.window_offset = 0;
				}
			}
		}
	}
}
