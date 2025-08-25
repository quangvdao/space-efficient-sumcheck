use ark_ff::Field;
use crate::interpolation::field_mul_small::FieldMulSmall;

use crate::{
	messages::VerifierMessages,
	multilinear_product::{StreamingEvalProductProver, StreamingEvalProductProverConfig, TimeProductProver, SchedulingParams, compute_eval_product_schedule},
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


		// Use shared scheduling logic for EvalProduct algorithm
		debug_assert!(D >= 2, "StreamingEvalProductProver requires D >= 2");
		
		let scheduling_params = SchedulingParams {
			d: D,
			num_variables,
			num_stages,
		};
		let state_comp_set = compute_eval_product_schedule(&scheduling_params);
		
		// For compatibility, set last_round_phase1 to 0 (no special early phase for D>=2)
		let last_round_phase1: usize = 0;
		let last_round: usize = state_comp_set.iter().next_back().copied().unwrap_or(num_variables);

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

		let vsbw_prover = TimeProductProver::<F, S, D> {
			claim: prover_config.claim,
			current_round: 0,
			evaluations: std::array::from_fn(|_| None),
			streams: None,
			num_variables: num_variables - last_round + 1,
		};
		
		Self {
			claim: prover_config.claim,
			current_round: 0,
			streams,
			stream_iterators,
			num_stages,
			num_variables,
			last_round_phase1,
			verifier_messages: VerifierMessages::new(&vec![]),
			verifier_messages_round_comp: VerifierMessages::new(&vec![]),
			stage_size,
			prev_table_round_num: 0,
			prev_table_size: 0,
			state_comp_set,
			switched_to_vsbw: false,
			vsbw_prover,
			current_window_size: 0,
			window_offset: 0,
			reduced_grid: None,
			reduced_shape: Vec::new(),
		}
	}

	fn next_message(
		&mut self,
		verifier_message: Self::VerifierMessage,
		_claim_sum: F,
	) -> Self::ProverMessage {
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
		println!("DEBUG Streaming: next_message round={}, sums={:?}", self.current_round, sums);

		self.current_round += 1;
		if self.switched_to_vsbw {
			self.vsbw_prover.current_round += 1;
		}
		Some(sums)
	}
}
