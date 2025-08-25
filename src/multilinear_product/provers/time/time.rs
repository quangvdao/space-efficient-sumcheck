use ark_ff::Field;
use ark_std::vec::Vec;

use crate::streams::Stream;

pub struct TimeProductProver<F: Field, S: Stream<F>, const D: usize> {
    pub claim: F,
    pub current_round: usize,
    pub evaluations: [Option<Vec<F>>; D],
    pub streams: Option<[S; D]>,
    pub num_variables: usize,
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
    pub fn vsbw_evaluate(&self) -> Vec<F> {
        // Message shape: [1, 2, ..., D-1, ∞] (for D=1: [1])
        let num_extras = if D > 2 { D - 2 } else { 0 };
        let mut sums: Vec<F> = vec![F::ZERO; D];

        // Calculate the bitmask for the number of free variables
        let bitmask: usize = 1 << (self.num_free_variables() - 1);

        // Determine the length of evaluations to iterate through
        let evaluations_len = match &self.evaluations[0] {
            Some(evaluations) => evaluations.len(),
            None => match &self.streams {
                Some(streams) => 1usize << streams[0].num_variables(),
                None => panic!("Both streams and evaluations cannot be None"),
            },
        };

        // Iterate through evaluations
        for i in 0..(evaluations_len / 2) {
            let mut prod_g1: F = F::ONE; // product at x = 1
            let mut prod_leading: F = F::ONE; // product at ∞ (top coeffs)
            let mut prod_extras_arr: [F; 30] = [F::ONE; 30]; // supports up to D=32

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
                
                // Debug tiny cases
                if self.num_free_variables() <= 2 && i < 2 && j < 1 {
                    println!("DEBUG TimeProver: i={}, j={}, pos0={}, pos1={}, v0={:?}, v1={:?}", 
                        i, j, i, i | bitmask, v0, v1);
                }
                
                // values at 1 and ∞
                let diff = v1 - v0;
                prod_leading *= diff; // ∞
                prod_g1 *= v1;
                // extra nodes z = 2..D-1 evaluated iteratively: val(z) = v1 + (z-1)*diff
                if num_extras > 0 {
                    let mut val = v1 + diff;
                    prod_extras_arr[0] *= val;
                    for k in 1..num_extras {
                        val += diff;
                        prod_extras_arr[k] *= val;
                    }
                }
            }
            if D == 1 {
                sums[0] += prod_g1; // [1]
            } else {
                sums[0] += prod_g1; // 1
                for k in 0..num_extras { sums[1 + k] += prod_extras_arr[k]; } // 2..D-1
                sums[D - 1] += prod_leading; // ∞
            }
        }

        sums
    }
    pub fn vsbw_reduce_evaluations(&mut self, verifier_message: F) {
        for i in 0..D {
            // Clone or initialize the evaluations vector
            let mut evaluations = match &self.evaluations[i] {
                Some(evaluations) => evaluations.clone(),
                None => match &self.streams {
                    Some(streams) => vec![
                        F::ZERO;
                        (1usize << streams[i].num_variables()) / 2
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
                let e0 = match &self.evaluations[i] {
                    None => match &self.streams {
                        Some(streams) => streams[i].evaluation(i0),
                        None => panic!("Both streams and evaluations cannot be None"),
                    },
                    Some(evaluations) => evaluations[i0],
                };
                let e1 = match &self.evaluations[i] {
                    None => match &self.streams {
                        Some(streams) => streams[i].evaluation(i1),
                        None => panic!("Both streams and evaluations cannot be None"),
                    },
                    Some(evaluations) => evaluations[i1],
                };
                // Smarter reduction: e0 + r * (e1 - e0) saves one multiplication
                let diff = e1 - e0;
                evaluations[i0] = e0 + diff * verifier_message;
            }

            // Truncate the evaluations vector to the correct length
            evaluations.truncate(evaluations_len);

            // Update the internal state with the new evaluations vector
            self.evaluations[i] = Some(evaluations.clone());
        }
    }
}
