use crate::{
	messages::VerifierMessages,
	multilinear_product::TimeProductProver,
	order_strategy::SignificantBitOrder,
	streams::{Stream, StreamIterator},
	interpolation::{field_mul_small::FieldMulSmall, LagrangePolynomial},
};
use crate::interpolation::multivariate::{compute_strides, multivariate_product_evaluations};
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

/// EvalProductProductProver is a placeholder for an evaluation-basis variant of the
/// product prover. It mirrors the public interface of Blendy but defers core
/// logic to future work. For now it routes phase-3 to the existing time prover
/// and stubs phase-1/2 computations.
pub struct StreamingEvalProductProver<F: Field, S: Stream<F>, const D: usize> {
	pub claim: F,
	pub current_round: usize,
	pub streams: [S; D],
	pub stream_iterators: [StreamIterator<F, S, SignificantBitOrder>; D],
	pub num_stages: usize,
	pub num_variables: usize,
	pub windows: Vec<usize>,
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
	pub current_window_idx: usize,
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

	/// Collapse axis 0 by evaluating its A-nodes line at r (univariate extrapolation) and
	/// replacing the grid with one fewer axis.
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

	fn build_window_grid(&mut self, omega: usize)
	where
		F: FieldMulSmall,
	{
		// Fast path: omega == 1 can be computed directly in the univariate domain
		if omega == 1 {
			let n = self.num_variables;
			let bitmask: usize = 1 << (n - 1); // split on the current round (MSB)
			let half = 1usize << (n - 1);
			let mut sums: Vec<F> = vec![F::ZERO; if D > 1 { D } else { 1 }];
			for i in 0..half {
				let mut prod_g1: F = F::ONE;
				let mut prod_leading: F = F::ONE;
				let mut prod_extras: [F; 30] = [F::ONE; 30];
				let num_extras = if D > 2 { D - 2 } else { 0 };
				for j in 0..D {
					let v0 = self.streams[j].evaluation(i);
					let v1 = self.streams[j].evaluation(i | bitmask);
					let diff = v1 - v0;
					prod_leading *= diff;
					prod_g1 *= v1;
					if num_extras > 0 {
						let mut val = v1 + diff; // g(2)
						prod_extras[0] *= val;
						for k in 1..num_extras {
							val += diff; // advance to g(3), g(4), ...
							prod_extras[k] *= val;
						}
					}
				}
				if D == 1 {
					sums[0] += prod_g1;
				} else {
					sums[0] += prod_g1; // g(1)
					for k in 0..num_extras { sums[1 + k] += prod_extras[k]; }
					sums[D - 1] += prod_leading; // ∞
				}
			}
			// Materialize a single-axis grid line on U_D: [1, 2, ..., D-1, D=?, ∞]
			self.reduced_shape = vec![D + 1];
			let mut line: Vec<F> = vec![F::ZERO; D + 1];
			if D == 1 {
				line[0] = sums[0];
			} else {
				for z in 1..D { line[z - 1] = sums[z - 1]; }
				line[D] = sums[D - 1]; // ∞
			}
			self.reduced_grid = Some(line);
			return;
		}

		// Derive dimensions
		let j_prime = self.current_round + 1; // window starts at current round (1-indexed)
		let x_num_vars = if j_prime == 0 { 0 } else { j_prime - 1 };
		let b_num_vars = self.num_variables + 1 - j_prime - omega;
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
			let mut polys_01: Vec<Vec<F>> = Vec::with_capacity(D);
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
				polys_01.push(window_vals_01);
			}
			let prod = multivariate_product_evaluations::<F>(omega, &polys_01, D);
			let grid = self.reduced_grid.as_mut().expect("grid should be allocated");
			for i in 0..grid.len() { grid[i] += prod[i]; }
		}
	}

	/// Round computation: use reduced_grid to produce [g(1), g(2), ..., g(D-1), g(∞)].
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
		// Now cur is a single line along axis 0 of length D+1 in U_D order [1..D, ∞]
		for z in 1..D { out[z - 1] = cur[z - 1]; }
		out[D - 1] = cur[D];
		out
	}

	/// Stage/window management: initialize grid at window start; collapse after binding.
	pub fn compute_state(&mut self) {
		// Tail switch behavior maintained
		let j = self.current_round + 1;
		let p = self.state_comp_set.contains(&j);
		let is_largest = self.state_comp_set.range((j + 1)..).next().is_none();
		if p && is_largest {
			let num_variables_new = self.num_variables - j + 1;
			self.switched_to_vsbw = true;
			self.stream_iterators.iter_mut().for_each(|it| it.reset());
			let side = 1 << num_variables_new;
			let mut evals: [Vec<F>; D] = std::array::from_fn(|_| vec![F::ZERO; side]);
			for t in 0..D { self.vsbw_prover.evaluations[t] = Some(std::mem::take(&mut evals[t])); }
			return;
		}
		if self.switched_to_vsbw {
			let verifier_message = self.verifier_messages.messages[self.current_round - 1];
			self.vsbw_prover.vsbw_reduce_evaluations(verifier_message);
			return;
		}

		// Window-grid flow
		if self.reduced_grid.is_none() {
			// Start of a new window: allocate zero grid with shape (D+1)^ω and populate from streams
			if self.current_window_idx < self.windows.len() {
				let omega = self.windows[self.current_window_idx];
				self.reduced_shape = core::iter::repeat(D + 1).take(omega).collect();
				let total = self.reduced_shape.iter().copied().fold(1usize, |acc, s| acc * s);
				self.reduced_grid = Some(vec![F::ZERO; total]);
				self.window_offset = 0;
				self.build_window_grid(omega);
			} else {
				// No more windows; remain in per-round streaming (not yet implemented)
			}
		} else {
			// Collapse axis 0 with the last challenge r to advance within the window
			if self.current_round > 0 && self.window_offset < self.reduced_shape.len() {
				let r = self.verifier_messages.messages[self.current_round - 1];
				self.collapse_axis_at_point(r);
				// If finished window, move to next window
				if self.reduced_shape.is_empty() {
					self.reduced_grid = None;
					self.current_window_idx += 1;
					self.window_offset = 0;
				}
			}
		}
	}
}
