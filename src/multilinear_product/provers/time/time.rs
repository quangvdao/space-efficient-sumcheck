use ark_ff::Field;
use ark_std::vec::Vec;

use crate::streams::Stream;

pub struct TimeProductProver<F: Field, S: Stream<F>, const D: usize> {
    pub claim: F,
    pub current_round: usize,
    pub evaluations: [Option<Vec<F>>; D],
    pub streams: Option<[S; D]>,
    pub num_variables: usize,
    pub inverse_two_pow_d: F,
}

impl<'a, F: Field, S: Stream<F>, const D: usize> TimeProductProver<F, S, D> {
    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }
    pub fn num_free_variables(&self) -> usize {
        self.num_variables - self.current_round
    }
    /*
     * Note in evaluate() there's an optimization for the first round where we read directly
     * from the streams (instead of the tables), which reduces max memory usage by 1/2
     */
    pub fn vsbw_evaluate(&self) -> (F, F, F) {
        // Initialize accumulators
        let mut sum_0 = F::ZERO;
        let mut sum_1 = F::ZERO;
        let mut sum_half = F::ZERO;

        // Calculate the bitmask for the number of free variables
        let bitmask: usize = 1 << (self.num_free_variables() - 1);

        // Determine the length of evaluations to iterate through
        let evaluations_len = match &self.evaluations[0] {
            Some(evaluations) => evaluations.len(),
            None => match &self.streams {
                Some(streams) => 2usize.pow(streams[0].num_variables() as u32),
                None => panic!("Both streams and evaluations cannot be None"),
            },
        };

        // Iterate through evaluations
        for i in 0..(evaluations_len / 2) {
            let mut prod0 = F::ONE;
            let mut prod1 = F::ONE;
            let mut prod_half = F::ONE;

            for j in 0..D {
                let v0 = match &self.evaluations[j] {
                    None => match &self.streams {
                        Some(streams) => streams[j].evaluation(i),
                        None => panic!("Both streams and evaluations cannot be None"),
                    },
                    Some(evals) => evals[i],
                };
                let v1 = match &self.evaluations[j] {
                    None => match &self.streams {
                        Some(streams) => streams[j].evaluation(i | bitmask),
                        None => panic!("Both streams and evaluations cannot be None"),
                    },
                    Some(evals) => evals[i | bitmask],
                };
                prod0 *= v0;
                prod1 *= v1;
                prod_half *= v0 + v1;
            }

            sum_0 += prod0;
            sum_1 += prod1;
            sum_half += prod_half;
        }

        sum_half = sum_half * self.inverse_two_pow_d;

        (sum_0, sum_1, sum_half)
    }
    pub fn vsbw_reduce_evaluations(&mut self, verifier_message: F, verifier_message_hat: F) {
        for i in 0..D {
            // Clone or initialize the evaluations vector
            let mut evaluations = match &self.evaluations[i] {
                Some(evaluations) => evaluations.clone(),
                None => match &self.streams {
                    Some(streams) => vec![
                        F::ZERO;
                        2usize.pow(streams[i].num_variables().try_into().unwrap())
                            / 2
                    ],
                    None => panic!("Both streams and evaluations cannot be None"),
                },
            };

            // Determine the length of evaluations to iterate through
            let evaluations_len = match &self.evaluations[i] {
                Some(evaluations) => evaluations.len() / 2,
                None => evaluations.len(),
            };

            // Calculate what bit needs to be set to index the second half of the last round's evaluations
            let setbit: usize = 1 << self.num_free_variables();

            // Iterate through pairs of evaluations
            for i0 in 0..evaluations_len {
                let i1 = i0 | setbit;

                // Get point evaluations for indices i0 and i1
                let point_evaluation_i0 = match &self.evaluations[i] {
                    None => match &self.streams {
                        Some(streams) => streams[i].evaluation(i0),
                        None => panic!("Both streams and evaluations cannot be None"),
                    },
                    Some(evaluations) => evaluations[i0],
                };
                let point_evaluation_i1 = match &self.evaluations[i] {
                    None => match &self.streams {
                        Some(streams) => streams[i].evaluation(i1),
                        None => panic!("Both streams and evaluations cannot be None"),
                    },
                    Some(evaluations) => evaluations[i1],
                };
                // Update the i0-th evaluation based on the reduction operation
                evaluations[i0] = point_evaluation_i0 * verifier_message_hat
                    + point_evaluation_i1 * verifier_message;
            }

            // Truncate the evaluations vector to the correct length
            evaluations.truncate(evaluations_len);

            // Update the internal state with the new evaluations vector
            self.evaluations[i] = Some(evaluations.clone());
        }
    }
}
