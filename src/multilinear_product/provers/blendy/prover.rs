use ark_ff::Field;

use crate::{
    messages::VerifierMessages,
    multilinear_product::{BlendyProductProver, BlendyProductProverConfig, TimeProductProver, SchedulingParams, compute_cross_product_schedule},
    order_strategy::SignificantBitOrder,
    prover::Prover,
    streams::{Stream, StreamIterator},
};

impl<F: Field, S: Stream<F>, const D: usize> Prover<F> for BlendyProductProver<F, S, D> {
    type ProverConfig = BlendyProductProverConfig<F, S>;
    type ProverMessage = Option<Vec<F>>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        let num_variables: usize = prover_config.num_variables;
        let num_stages: usize = prover_config.num_stages;
        let stage_size: usize = num_variables / num_stages;

        // Use shared scheduling logic for CrossProduct algorithm
        debug_assert!(D >= 2, "BlendyProductProver requires D >= 2");
        
        let scheduling_params = SchedulingParams {
            d: D,
            num_variables,
            num_stages,
        };
        let state_comp_set = compute_cross_product_schedule(&scheduling_params);
        
        // For compatibility, set last_round_phase1 to 0 (no special early phase for D>=2)
        let last_round_phase1: usize = 0;

        let last_round: usize = state_comp_set.iter().next_back().copied().unwrap_or(num_variables + 1);
        let vsbw_prover = TimeProductProver::<F, S, D> {
            claim: prover_config.claim,
            current_round: 0,
            evaluations: std::array::from_fn(|_| None),
            streams: None,
            num_variables: if last_round <= num_variables { num_variables - last_round + 1 } else { 0 },
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
        // return the BlendyProver instance
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
            x_table: vec![],
            y_table: vec![],
            j_prime_table: vec![],
            partial_tables: None,
            j_prime_table_flat: None,
            stage_size,
            prev_table_round_num: 0,
            prev_table_size: 0,
            state_comp_set,
            switched_to_vsbw: false,
            vsbw_prover,
        }
    }

    fn next_message(&mut self, verifier_message: Self::VerifierMessage, _claim_sum: F) -> Self::ProverMessage {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if !self.is_initial_round() {
            // this holds everything
            self.verifier_messages
                .receive_message(verifier_message.unwrap());
            // this holds the randomness for between state computation r2
            self.verifier_messages_round_comp
                .receive_message(verifier_message.unwrap());
        }

        self.init_round_vars();

        self.compute_state();

        let sums: Vec<F> = self.compute_round();

        // Increment the round counter
        self.current_round += 1;
        if self.switched_to_vsbw {
            self.vsbw_prover.current_round += 1;
        }
        // Return the computed polynomial sums
        Some(sums)
    }
}

// #[cfg(test)]
// mod tests {
//     use ark_poly::multivariate::{SparsePolynomial, SparseTerm};

//     use crate::{
//         multilinear_product::{BlendyProductProver, BlendyProductProverConfig},
//         order_strategy::SignificantBitOrder,
//         prover::{ProductProverConfig, Prover},
//         streams::{multivariate_product_claim, MemoryStream, Stream},
//         tests::{
//             multilinear_product::{BasicProductProver, BasicProductProverConfig},
//             polynomials::Polynomial,
//             BenchStream, F64,
//         },
//         ProductSumcheck,
//     };

//     // the stream has to be in SigBit order for this to work
//     // #[test]
//     // fn parity_with_basic_prover() {
//     //     consistency_test::<F64, BenchStream<F64>, BlendyProductProver<F64, BenchStream<F64>>>();
//     // }

//     #[test]
//     fn consistency_test_with_next_iterator() {
//         // // get evals in lexicographic order
//         // let num_variables = 8;
//         // let s_tmp: BenchStream<F64> = BenchStream::<F64>::new(num_variables).into();
//         // let mut evals: Vec<F64> = Vec::with_capacity(1 << num_variables);
//         // for i in 0..(1 << num_variables) {
//         //     evals.push(s_tmp.evaluation(i));
//         // }

//         // // create the stream in SigBit order
//         // let s: MemoryStream<F64> =
//         //     MemoryStream::new_from_lex::<SignificantBitOrder>(evals.clone()).into();
//         // let claim: F64 = multivariate_product_claim(vec![s.clone(), s.clone()]);

//         // // get transcript from Blendy prover (message shape may differ across implementations)
//         // let prover_transcript: ProductSumcheck<F64, 2> = ProductSumcheck::<F64, 2>::prove::<
//         //     MemoryStream<F64>,
//         //     BlendyProductProver<F64, MemoryStream<F64>, 2>,
//         // >(
//         //     &mut Prover::<F64>::new(BlendyProductProverConfig::default(
//         //         claim,
//         //         num_variables,
//         //         vec![s.clone(), s],
//         //     )),
//         //     &mut ark_std::test_rng(),
//         // );

//         // get transcript from SanityProver
//         // let p: SparsePolynomial<F64, SparseTerm> =
//         //     <SparsePolynomial<F64, SparseTerm> as Polynomial<F64>>::from_hypercube_evaluations(
//         //         evals,
//         //     );
//         // let mut sanity_prover = BasicProductProver::<F64>::new(BasicProductProverConfig::new(
//         //     claim.clone(),
//         //     num_variables,
//         //     p.clone(),
//         //     p,
//         // ));
//         // let sanity_prover_transcript = ProductSumcheck::<F64, 2>::prove::<
//         //     MemoryStream<F64>,
//         //     BasicProductProver<F64>,
//         // >(&mut sanity_prover, &mut ark_std::test_rng());
//         // Round 1: both must satisfy g_0(0)+g_0(1) = claim (g_0(1) may be derived in prover transcript)
//         // let g0_0 = prover_transcript.prover_messages[0][0];
//         // let g0_1 = claim - g0_0;
//         // let s0 = sanity_prover_transcript.prover_messages[0][0];
//         // assert_eq!(s0 + (claim - s0), claim);
//     }
// }
