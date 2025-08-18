use ark_ff::Field;
use ark_std::{rand::Rng, vec::Vec};

use crate::{prover::Prover, streams::Stream};

#[derive(Debug)]
pub struct Sumcheck<F: Field> {
    pub prover_messages: Vec<(F, F)>,
    pub verifier_messages: Vec<F>,
}

impl<F: Field> Sumcheck<F> {
    pub fn prove<S, P>(prover: &mut P, rng: &mut impl Rng) -> Self
    where
        S: Stream<F>,
        P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F)>>,
    {
        // Initialize vectors to store prover and verifier messages
        let mut prover_messages: Vec<(F, F)> = vec![];
        let mut verifier_messages: Vec<F> = vec![];

        // Run the protocol
        let mut verifier_message: Option<F> = None;
        while let Some(message) = prover.next_message(verifier_message) {
            // Handle how to proceed
            prover_messages.push(message);
            let r = F::rand(rng);
            verifier_message = Some(r);
            verifier_messages.push(r);
        }

        // Return a Sumcheck struct with the collected messages and acceptance status
        Sumcheck {
            prover_messages,
            verifier_messages,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Sumcheck;
    use crate::{
        multilinear::{BlendyProver, BlendyProverConfig, TimeProver},
        prover::{Prover, ProverConfig},
        tests::{BenchStream, F19},
    };

    #[test]
    fn algorithm_consistency() {
        // take an evaluation stream
        let evaluation_stream: BenchStream<F19> = BenchStream::new(20);
        let claim = evaluation_stream.claimed_sum;
        // initialize the provers
        let mut blendy_k3_prover = BlendyProver::<F19, BenchStream<F19>>::new(
            BlendyProverConfig::new(claim, 3, 20, evaluation_stream.clone()),
        );
        let mut time_prover = TimeProver::<F19, BenchStream<F19>>::new(<TimeProver<
            F19,
            BenchStream<F19>,
        > as Prover<F19>>::ProverConfig::default(
            claim,
            20,
            evaluation_stream,
        ));
        // run them and get the transcript
        let blendy_prover_transcript = Sumcheck::<F19>::prove::<
            BenchStream<F19>,
            BlendyProver<F19, BenchStream<F19>>,
        >(&mut blendy_k3_prover, &mut ark_std::test_rng());
        let time_prover_transcript = Sumcheck::<F19>::prove::<
            BenchStream<F19>,
            TimeProver<F19, BenchStream<F19>>,
        >(&mut time_prover, &mut ark_std::test_rng());
        // ensure the transcript is identical
        assert_eq!(
            time_prover_transcript.prover_messages,
            blendy_prover_transcript.prover_messages
        );
    }
}
