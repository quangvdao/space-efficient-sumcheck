use ark_ff::Field;
use ark_std::vec::Vec;

use crate::streams::Stream;
use crate::interpolation::univariate::product_eval_univariate_accumulate;
use crate::interpolation::field_mul_small::FieldMulSmall;

pub struct ImprovedTimeProductProver<F: Field, S: Stream<F>, const D: usize> {
    pub claim: F,
    pub current_round: usize,
    pub evaluations: [Option<Vec<F>>; D],
    pub streams: Option<[S; D]>,
    pub num_variables: usize,
}

impl<'a, F: FieldMulSmall, S: Stream<F>, const D: usize> ImprovedTimeProductProver<F, S, D> {
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
    pub fn toom_evaluate(&self) -> Vec<F> {
        // Message shape: [1, 2, ..., D-1, ∞]
        let mut sums: Vec<F> = vec![F::ZERO; D];

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

        // Iterate through evaluations (each i corresponds to a slice over remaining variables)
        for i in 0..(evaluations_len / 2) {
            // Gather D linear polynomials for this slice
            let mut pairs: [(F, F); D] = [(F::ZERO, F::ZERO); D];
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
                pairs[j] = (v0, v1);
            }

            // Fused accumulation: avoid materializing the full table
            product_eval_univariate_accumulate::<F, D>(&pairs, &mut sums);
        }

        sums
    }
    pub fn reduce_evaluations(&mut self, verifier_message: F) {
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
                // Smarter reduction: e0 + r * (e1 - e0) saves one multiplication
                let diff = point_evaluation_i1 - point_evaluation_i0;
                evaluations[i0] = point_evaluation_i0 + diff * verifier_message;
            }

            // Truncate the evaluations vector to the correct length
            evaluations.truncate(evaluations_len);

            // Update the internal state with the new evaluations vector
            self.evaluations[i] = Some(evaluations.clone());
        }
    }
}
