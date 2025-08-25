use ark_ff::Field;

use crate::{
    messages::VerifierMessages,
    prover::Prover,
    tests::multilinear_product::{BasicProductProver, BasicProductProverConfig},
};

impl<F: Field> Prover<F> for BasicProductProver<F> {
    type ProverConfig = BasicProductProverConfig<F>;
    type ProverMessage = Option<Vec<F>>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F {
        self.claim
    }

    fn new(prover_config: Self::ProverConfig) -> Self {
        Self {
            claim: prover_config.claim,
            current_round: 0,
            num_variables: prover_config.num_variables,
            p: prover_config.p,
            q: prover_config.q,
            verifier_messages: VerifierMessages::new(&vec![]),
        }
    }

    fn next_message(&mut self, verifier_message: Self::VerifierMessage, _claim_sum: F) -> Self::ProverMessage {
        // Ensure the current round is within bounds
        if self.current_round >= self.total_rounds() {
            return None;
        }

        if !self.is_initial_round() {
            self.verifier_messages
                .receive_message(verifier_message.unwrap());
        }

        let (g0, leading) = self.compute_round_components();

        // Increment the round counter
        self.current_round += 1;

        // Return the computed polynomial sums in node order [0, ∞]
        Some(ark_std::vec![g0, leading])
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        prover::Prover,
        streams::{multivariate_product_claim, MemoryStream},
        tests::{
            multilinear_product::{
                sanity_test_driver, BasicProductProver, BasicProductProverConfig,
            },
            polynomials::{four_variable_polynomial, four_variable_polynomial_evaluations},
            F19,
        },
    };
    #[test]
    fn sumcheck() {
        let s = MemoryStream::<F19>::new(four_variable_polynomial_evaluations());
        sanity_test_driver(&mut BasicProductProver::new(BasicProductProverConfig::new(
            multivariate_product_claim(vec![s.clone(), s]),
            4,
            four_variable_polynomial(),
            four_variable_polynomial(),
        )));
    }
}
