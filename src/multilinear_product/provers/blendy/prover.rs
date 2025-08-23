use ark_ff::Field;
use std::collections::BTreeSet;

use crate::{
    messages::VerifierMessages,
    multilinear_product::{BlendyProductProver, BlendyProductProverConfig, TimeProductProver},
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

        // Paper-aligned parameters
        // ell := floor(n / (d * k)) with d = D and k = num_stages
        let ell: usize = core::cmp::max(1, num_variables / (D * num_stages));
        let time_phase_end: usize = (D.saturating_sub(1)).saturating_mul(ell); // (d-1)*ell
        // Tail switch window: final ell/k rounds (at least 1)
        let tail_rounds: usize = core::cmp::max(1, ell / core::cmp::max(1, num_stages));
        let last_round_phase3: usize = num_variables.saturating_sub(tail_rounds);
        // Early rounds without passes: start immediately (j'=1), so no special phase1 for D>=2.
        // For D==1 we avoid passes entirely and stream/linear-time only.
        let last_round_phase1: usize = if D == 1 { num_variables } else { 0 };

        let state_comp_set: BTreeSet<usize> = {
            let mut set: BTreeSet<usize> = BTreeSet::new();
            if D >= 2 {
                // Time-constrained phase: j' = ceil(eta^t) where eta = d/(d-1)
                // Generate by recurrence: j_{t+1} = ceil(eta * j_t) with j_0 = 1
                // ceil(a/b * x) = (a*x + b - 1)/b
                let num: usize = D; // numerator of eta
                let den: usize = D - 1; // denominator of eta
                let mut j_start: usize = 1;
                while j_start <= time_phase_end && j_start <= last_round_phase3 {
                    set.insert(j_start);
                    let next = (num.saturating_mul(j_start) + (den - 1)) / den;
                    if next <= j_start { break; }
                    j_start = next;
                }
                // Space-constrained phase: passes every ell rounds when j >= (d-1)*ell
                if ell > 0 {
                    let m = if time_phase_end == 0 { 1 } else { (time_phase_end + ell - 1) / ell };
                    let mut j_space = m.saturating_mul(ell);
                    if j_space == 0 { j_space = ell; }
                    while j_space <= last_round_phase3 {
                        set.insert(j_space);
                        let (next, overflow) = j_space.overflowing_add(ell);
                        if overflow { break; }
                        j_space = next;
                    }
                }
                // Ensure we have at least one stage start
                if set.is_empty() { set.insert(1); }
            }
            set
        };
        if D >= 2 { assert!(state_comp_set.len() > 0); }

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

    fn next_message(&mut self, verifier_message: Self::VerifierMessage) -> Self::ProverMessage {
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
