use crate::{
	messages::VerifierMessages,
	multilinear_product::TimeProductProver,
	order_strategy::SignificantBitOrder,
	streams::{Stream, StreamIterator},
};
use ark_ff::Field;
use ark_std::vec::Vec;
use std::collections::BTreeSet;

/// EvalToomProductProver is a placeholder for an evaluation-basis variant of the
/// product prover. It mirrors the public interface of Blendy but defers core
/// logic to future work. For now it routes phase-3 to the existing time prover
/// and stubs phase-1/2 computations.
pub struct EvalToomProductProver<F: Field, S: Stream<F>, const D: usize> {
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
}

impl<F: Field, S: Stream<F>, const D: usize> EvalToomProductProver<F, S, D> {
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

	/// Stubbed round computation. For now, it returns zeros with length D+1,
	/// matching the node set used by Blendy (0, 1, 1/2, 2, ..., D-1).
	pub fn compute_round(&mut self) -> Vec<F> {
		if self.switched_to_vsbw {
			return self.vsbw_prover.vsbw_evaluate();
		}
		let mut result = Vec::with_capacity(D + 1);
		for _ in 0..(D + 1) {
			result.push(F::ZERO);
		}
		result
	}

	/// Stubbed state computation. Prepares for switching to the time-based
	/// prover on the last segment; otherwise, does nothing for now.
	pub fn compute_state(&mut self) {
		let j = self.current_round + 1;
		let p = self.state_comp_set.contains(&j);
		let is_largest = self.state_comp_set.range((j + 1)..).next().is_none();

		if p && is_largest {
			let num_variables_new = self.num_variables - j + 1;
			self.switched_to_vsbw = true;

			self.stream_iterators
				.iter_mut()
				.for_each(|stream_it| stream_it.reset());

			let side = 1 << num_variables_new;
			let mut evals: [Vec<F>; D] = std::array::from_fn(|_| vec![F::ZERO; side]);
			// Leave evals as zeros for now; future work will compute evaluation-basis tables.
			for t in 0..D {
				self.vsbw_prover.evaluations[t] = Some(std::mem::take(&mut evals[t]));
			}
		} else if self.switched_to_vsbw {
			let verifier_message = self.verifier_messages.messages[self.current_round - 1];
			self.vsbw_prover
				.vsbw_reduce_evaluations(verifier_message, F::ONE - verifier_message);
		}
	}
}
