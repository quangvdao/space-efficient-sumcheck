use crate::interpolation::field_mul_small::FieldMulSmall;

use crate::{
    multilinear_product::{ImprovedTimeProductProver, ImprovedTimeProductProverConfig},
    prover::Prover,
    streams::Stream,
};

impl<F: FieldMulSmall, S: Stream<F>, const D: usize> Prover<F> for ImprovedTimeProductProver<F, S, D> {
    type ProverConfig = ImprovedTimeProductProverConfig<F, S>;
    type ProverMessage = Option<Vec<F>>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        let num_variables = prover_config.num_variables;
        let streams_vec = prover_config.streams;
        assert!(streams_vec.len() == D);
        let streams_arr: [S; D] = streams_vec
            .try_into()
            .ok()
            .expect("streams length must equal D");
        Self {
            claim: prover_config.claim,
            current_round: 0,
            evaluations: None,
            streams: Some(streams_arr),
            num_variables,
        }
    }

    fn next_message(&mut self, verifier_message: Option<F>, _claim_sum: F) -> Option<Vec<F>> {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        // If it's not the first round, reduce the evaluations table
        if self.current_round != 0 {
            // update the evaluations table by absorbing leftmost variable assigned to verifier_message
            self.reduce_evaluations(verifier_message.unwrap());
        }

        // evaluate using recursive toom-cook-based product
        let sums = self.toom_evaluate();

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial evaluations as a vector
        return Some(sums);
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{multilinear_product::consistency_test, BenchStream, BN254};

    use super::ImprovedTimeProductProver;

    #[test]
    fn parity_with_basic_prover() {
        consistency_test::<BN254, BenchStream<BN254>, ImprovedTimeProductProver<BN254, BenchStream<BN254>, 2>, 2>();
    }
}
