use crate::{
	messages::VerifierMessages,
	multilinear_product::TimeProductProver,
	order_strategy::SignificantBitOrder,
	streams::{Stream, StreamIterator},
	interpolation::{field_mul_small::FieldMulSmall, LagrangePolynomial},
};
use crate::interpolation::multivariate::{compute_strides, multivariate_extrapolate, multivariate_product_evaluations};
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
		// Derive dimensions
		let j_prime = self.current_round + 1; // window starts at current round (1-indexed)
		let x_num_vars = if j_prime == 0 { 0 } else { j_prime - 1 };
		let b_num_vars = self.num_variables + 1 - j_prime - omega;
		assert!(b_num_vars as isize >= 0, "invalid window dims");
		// Precompute Lagrange weights for prefix over x_num_vars
		let mut lag_polys: Vec<F> = vec![F::ONE; 1usize << x_num_vars];
		if x_num_vars > 0 {
			let mut sequential_lag_poly = LagrangePolynomial::<F, SignificantBitOrder>::new(&self.verifier_messages);
			for (x_index, _) in Hypercube::<SignificantBitOrder>::new(x_num_vars) {
				lag_polys[x_index] = sequential_lag_poly.next().unwrap();
			}
		}
		// No explicit node construction; will use canonical extrapolation
		// Reset streams
		self.stream_iterators.iter_mut().for_each(|it| it.reset());
		// Output length per grid computed later when needed
		// Iterate over trailing b
		for (_, _) in Hypercube::<SignificantBitOrder>::new(b_num_vars) {
			// per-poly window slices on {0,1}^ω
			let mut polys_ad: Vec<Vec<F>> = Vec::with_capacity(D);
			for j in 0..D {
				let mut window_vals_01: Vec<F> = vec![F::ZERO; 1usize << omega];
				for (b_prime_index, _) in Hypercube::<SignificantBitOrder>::new(omega) {
					let mut sum = F::ZERO;
					for (x_index, _) in Hypercube::<SignificantBitOrder>::new(x_num_vars) {
						let w = lag_polys[x_index];
						let v = self.stream_iterators[j].next().unwrap();
						sum += w * v;
					}
					window_vals_01[b_prime_index] = sum;
				}
				// Lift {0,1}^ω to U_1^ω by replacing along each axis: [0,1] -> [1, ∞] = [f(1), f(1)-f(0)]
				let mut cur = window_vals_01.clone();
				let shape: Vec<usize> = core::iter::repeat(2usize).take(omega).collect();
				for axis in 0..omega {
					let strides = compute_strides(&shape);
					let mut next = vec![F::ZERO; cur.len()];
					let mut num_slices = 1usize;
					for (i, s) in shape.iter().enumerate() { if i != axis { num_slices *= *s; } }
					for slice_idx in 0..num_slices {
						let mut rem = slice_idx;
						let mut base = 0usize;
						for i in 0..omega {
							if i == axis { continue; }
							let si = shape[i];
							let coord = rem % si;
							rem /= si;
							base += coord * strides[i];
						}
						let f0 = cur[base + 0 * strides[axis]];
						let f1 = cur[base + 1 * strides[axis]];
						next[base + 0 * strides[axis]] = f1;             // 1
						next[base + 1 * strides[axis]] = f1 - f0;         // ∞
					}
					cur = next;
				}
				let window_vals_ad = multivariate_extrapolate::<F>(omega, 1, D, &cur);
				polys_ad.push(window_vals_ad);
			}
			let prod = multivariate_product_evaluations::<F>(omega, &polys_ad, D);
			// accumulate into reduced grid
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
		for z in 1..D { // fill 1..D-1
			let idx0 = z - 1;
			out[z - 1] = self.sum_over_remaining_axes_at_axis0_index(idx0);
		}
		// last slot: ∞
		out[D - 1] = self.sum_over_remaining_axes_at_axis0_index(D);
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
