use ark_ff::Field;

use crate::{
    messages::VerifierMessages,
    multilinear_product::{SpaceProductProver, SpaceProductProverConfig},
    order_strategy::SignificantBitOrder,
    prover::Prover,
    streams::{Stream, StreamIterator},
};

impl<F: Field, S: Stream<F>, const D: usize> Prover<F> for SpaceProductProver<F, S, D> {
    type ProverConfig = SpaceProductProverConfig<F, S>;
    type ProverMessage = Option<Vec<F>>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        let streams_vec = prover_config.streams;
        assert!(streams_vec.len() == D, "streams length must equal D");
        let iters_vec: Vec<StreamIterator<F, S, SignificantBitOrder>> = streams_vec
            .into_iter()
            .map(|s| StreamIterator::<F, S, SignificantBitOrder>::new(s))
            .collect();
        let stream_iterators: [StreamIterator<F, S, SignificantBitOrder>; D] = iters_vec
            .try_into()
            .ok()
            .expect("streams length must equal D");

        Self {
            claim: prover_config.claim,
            stream_iterators,
            verifier_messages: VerifierMessages::new(&vec![]),
            current_round: 0,
            num_variables: prover_config.num_variables,
        }
    }

    fn next_message(&mut self, verifier_message: Self::VerifierMessage, _claim_sum: F) -> Self::ProverMessage {
        // Ensure the current round is within bounds
        if self.current_round >= self.num_variables {
            return None;
        }

        // If it's not the first round, add the verifier message to verifier_messages
        if self.current_round != 0 {
            self.verifier_messages
                .receive_message(verifier_message.unwrap());
        }

        // evaluate using cty
        let sums: Vec<F> = self.cty_evaluate();

        // don't forget to increment the round
        self.current_round += 1;

        Some(sums)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear_product::SpaceProductProver,
        streams::MemoryStream,
        tests::{multilinear_product::sanity_test, F19},
    };

    #[test]
    fn sumcheck() {
        sanity_test::<F19, MemoryStream<F19>, SpaceProductProver<F19, MemoryStream<F19>, 2>>();
    }
}
