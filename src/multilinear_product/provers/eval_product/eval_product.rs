use crate::{
	messages::VerifierMessages,
	multilinear_product::TimeProductProver,
	order_strategy::SignificantBitOrder,
	streams::{Stream, StreamIterator},
};
use crate::interpolation::multivariate::{Node, univariate_extrapolate_nodes, multivariate_extrapolate_nodes, multivariate_product_evaluations_nodes_with_axes};
use crate::hypercube::Hypercube;
use ark_ff::Field;
use ark_std::vec::Vec;
use std::collections::BTreeSet;

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

impl<F: Field, S: Stream<F>, const D: usize> StreamingEvalProductProver<F, S, D> {
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

	#[inline]
	fn make_axis_nodes() -> Vec<Node<F>> {
		let mut nodes: Vec<Node<F>> = Vec::with_capacity(D + 1);
		for z in 1..=D { nodes.push(Node::Finite(F::from(z as u32))); }
		nodes.push(Node::Infinity);
		nodes
	}

	#[inline]
	fn compute_strides(shape: &[usize]) -> Vec<usize> {
		let v = shape.len();
		let mut strides = vec![1usize; v];
		for i in (0..v.saturating_sub(1)).rev() { strides[i] = strides[i + 1] * shape[i + 1]; }
		strides
	}

	/// Collapse axis 0 by evaluating its A-nodes line at r (univariate extrapolation) and
	/// replacing the grid with one fewer axis.
	fn collapse_axis_at_point(&mut self, r: F) {
		if self.reduced_grid.is_none() { return; }
		let grid = self.reduced_grid.as_ref().unwrap();
		let shape = &self.reduced_shape;
		if shape.is_empty() { return; }
		let strides = Self::compute_strides(shape);
		let axis_len = shape[0];
		let x_nodes = Self::make_axis_nodes();
		let y_nodes = [Node::Finite(r)];
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
			let vals = univariate_extrapolate_nodes::<F>(&x_nodes, &line, &y_nodes);
			out[out_idx] = vals[0];
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
		let strides = Self::compute_strides(shape);
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

	fn build_window_grid(&mut self, omega: usize) {
		// Derive dimensions
		let j_prime = self.current_round + 1; // window starts at current round (1-indexed)
		let x_num_vars = if j_prime == 0 { 0 } else { j_prime - 1 };
		let b_num_vars = self.num_variables + 1 - j_prime - omega;
		assert!(b_num_vars as isize >= 0, "invalid window dims");
		// Precompute Lagrange weights for prefix over x_num_vars
		let mut lag_polys: Vec<F> = vec![F::ONE; 1usize << x_num_vars];
		if x_num_vars > 0 {
			let mut sequential_lag_poly = crate::interpolation::LagrangePolynomial::<F, SignificantBitOrder>::new(&self.verifier_messages);
			for (x_index, _) in Hypercube::<SignificantBitOrder>::new(x_num_vars) {
				lag_polys[x_index] = sequential_lag_poly.next().unwrap();
			}
		}
		// Nodes for extrapolation
		let axes_k: Vec<Vec<Node<F>>> = (0..omega).map(|_| vec![Node::Finite(F::ZERO), Node::Finite(F::ONE)]).collect();
		let axis_d_nodes = Self::make_axis_nodes();
		let axes_d: Vec<Vec<Node<F>>> = (0..omega).map(|_| axis_d_nodes.clone()).collect();
		// Reset streams
		self.stream_iterators.iter_mut().for_each(|it| it.reset());
		// Output length per grid
		let out_len = (D + 1).pow(omega as u32);
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
				let window_vals_ad = multivariate_extrapolate_nodes::<F>(omega, &axes_k, &axes_d, &window_vals_01);
				polys_ad.push(window_vals_ad);
			}
			let prod = multivariate_product_evaluations_nodes_with_axes::<F>(omega, &polys_ad, &axes_d);
			// accumulate into reduced grid
			let grid = self.reduced_grid.as_mut().expect("grid should be allocated");
			for p in 0..out_len { grid[p] += prod[p]; }
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
