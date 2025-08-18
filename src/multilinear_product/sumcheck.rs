use ark_ff::Field;
use ark_std::{rand::Rng, vec::Vec};

use crate::{
    interpolation::LagrangePolynomial, order_strategy::GraycodeOrder, prover::Prover,
    streams::Stream,
};

#[derive(Debug, PartialEq)]
pub struct ProductSumcheck<F: Field, const D: usize> {
    pub prover_messages: Vec<Vec<F>>, // evaluations at interpolation nodes per round
    pub verifier_messages: Vec<F>,
}

impl<F: Field, const D: usize> ProductSumcheck<F, D> {
    pub fn prove<S, P>(prover: &mut P, rng: &mut impl Rng) -> Self
    where
        S: Stream<F>,
        P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<Vec<F>>>,
    {
        // Initialize vectors to store prover and verifier messages
        let mut prover_messages: Vec<Vec<F>> = vec![];
        let mut verifier_messages: Vec<F> = vec![];

        // Run the protocol
        let mut verifier_message: Option<F> = None;
        // Track previous target sum for deriving g(1) and evaluating next targets
        let mut prev_target_sum: F = prover.claim();
        while let Some(message) = prover.next_message(verifier_message) {
            debug_assert!(message.len() == D, "expected exactly D={} message entries", D);
            // New scheme only: message nodes are [0, ∞, 2, 3, ...] when D>1; when D==1: [0]
            debug_assert!(D == 1 || message.len() >= 2, "prover must send at least [g(0), g(∞)] when D>1");

            // Handle how to proceed
            // Persist message after acceptance decision
            prover_messages.push(message.clone());

            // Choose next verifier randomness and record it for the transcript
            let r = F::rand(rng);
            verifier_message = Some(r);
            verifier_messages.push(r);

            // Compute next target sum = g_k(r)
            let g0 = message[0];
            let g1 = prev_target_sum - g0;
            let next_target_sum: F = if D == 1 {
                // degree 1 per round over {0,1}
                g0 + r * (g1 - g0)
            } else {
                let leading = message[1];
                let extras = if message.len() > 2 { &message[2..] } else { &[] };
                LagrangePolynomial::<F,GraycodeOrder>::evaluate_from_infty_and_standard_nodes(
                    r,
                    leading,
                    g0,
                    g1,
                    extras,
                )
            };

            // Update target sum
            prev_target_sum = next_target_sum;
        }

        // Return a Sumcheck struct with the collected messages and acceptance status
        ProductSumcheck {
            prover_messages,
            verifier_messages,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        multilinear_product::TimeProductProver,
        tests::{multilinear_product::consistency_test, BenchStream, F64},
    };

    #[test]
    fn algorithm_consistency() {
        consistency_test::<F64, BenchStream<F64>, TimeProductProver<F64, BenchStream<F64>, 2>, 2>();
        // should take ordering of the stream
        // consistency_test::<F64, BenchStream<F64>, BlendyProductProver<F64, BenchStream<F64>>>();
    }
}
