use ark_ff::Field;
use crate::interpolation::field_mul_small::FieldMulSmall;
use std::collections::BTreeSet;

use crate::{
	messages::VerifierMessages,
	multilinear_product::{StreamingEvalProductProver, StreamingEvalProductProverConfig, TimeProductProver},
	order_strategy::SignificantBitOrder,
	prover::Prover,
	streams::{Stream, StreamIterator},
};

impl<F: Field + FieldMulSmall, S: Stream<F>, const D: usize> Prover<F>
	for StreamingEvalProductProver<F, S, D>
{
	type ProverConfig = StreamingEvalProductProverConfig<F, S>;
	type ProverMessage = Option<Vec<F>>;
	type VerifierMessage = Option<F>;

	fn claim(&self) -> F {
		self.claim
	}

	fn new(prover_config: Self::ProverConfig) -> Self {
		let num_variables: usize = prover_config.num_variables;
		let num_stages: usize = prover_config.num_stages;
		let stage_size: usize = num_variables / num_stages;
		let max_rounds_phase2: usize = num_variables.div_ceil(2 * num_stages);

		let last_round_phase1: usize = 2;
		let last_round_phase3: usize = num_variables - num_variables.div_ceil(num_stages);

		let state_comp_set: BTreeSet<usize> = {
			let mut current_round: usize = last_round_phase1 + 1;
			let mut state_comp_set: BTreeSet<usize> = BTreeSet::new();
			while current_round <= last_round_phase3 {
				state_comp_set.insert(current_round);
				current_round =
					std::cmp::min(current_round + max_rounds_phase2, current_round * 2 - 1);
				current_round = std::cmp::max(current_round, 2);
			}
			state_comp_set
		};
		assert!(state_comp_set.len() > 0);

		let last_round: usize = *state_comp_set.iter().next_back().unwrap();
		let vsbw_prover = TimeProductProver::<F, S, D> {
			claim: prover_config.claim,
			current_round: 0,
			evaluations: std::array::from_fn(|_| None),
			streams: None,
			num_variables: num_variables - last_round + 1,
		};

		let streams_vec = prover_config.streams;
		assert!(streams_vec.len() == D, "requires exactly D streams");
		let stream_iterators: [StreamIterator<F, S, SignificantBitOrder>; D] = streams_vec
			.iter()
			.cloned()
			.map(|s| StreamIterator::<F, S, SignificantBitOrder>::new(s))
			.collect::<Vec<_>>()
			.try_into()
			.ok()
			.expect("requires exactly D streams");
		let streams: [S; D] = streams_vec
			.try_into()
			.ok()
			.expect("requires exactly D streams");

		// Build window vector omegas from runtime num_stages (k)
		let mut windows: Vec<usize> = Vec::new();
		{
			let delta = ((D + 1) as f64).log2();
			let mut j_prime: usize = 0;
			// Time-constrained phase
			while j_prime < num_variables {
				let omega = ((j_prime as f64) / (delta - 1.0)).floor() as usize;
				let omega = omega.max(1);
				if j_prime + omega > num_variables { break; }
				windows.push(omega);
				j_prime += omega;
				// stop when window size reaches space phase target
				let omega_space = (num_variables / (num_stages * (delta as usize).max(1))).max(1);
				if omega >= omega_space { break; }
			}
			// Space-constrained phase
			let omega_space = (num_variables / (num_stages * (delta as usize).max(1))).max(1);
			while j_prime + omega_space <= num_variables {
				windows.push(omega_space);
				j_prime += omega_space;
			}
		}

		Self {
			claim: prover_config.claim,
			current_round: 0,
			streams,
			stream_iterators,
			num_stages,
			num_variables,
			windows,
			last_round_phase1,
			verifier_messages: VerifierMessages::new(&vec![]),
			verifier_messages_round_comp: VerifierMessages::new(&vec![]),
			stage_size,
			prev_table_round_num: 0,
			prev_table_size: 0,
			state_comp_set,
			switched_to_vsbw: false,
			vsbw_prover,
			current_window_idx: 0,
			window_offset: 0,
			reduced_grid: None,
			reduced_shape: Vec::new(),
		}
	}

	fn next_message(&mut self, verifier_message: Self::VerifierMessage) -> Self::ProverMessage {
		if self.current_round >= self.total_rounds() {
			return None;
		}

		if !self.is_initial_round() {
			self
				.verifier_messages
				.receive_message(verifier_message.unwrap());
			self
				.verifier_messages_round_comp
				.receive_message(verifier_message.unwrap());
		}

		self.init_round_vars();
		self.compute_state();
		let sums: Vec<F> = self.compute_round();

		self.current_round += 1;
		if self.switched_to_vsbw {
			self.vsbw_prover.current_round += 1;
		}
		Some(sums)
	}
}
