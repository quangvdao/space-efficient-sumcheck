use ark_ff::Field;
use ark_std::vec::Vec;

use crate::streams::Stream;
use crate::interpolation::univariate::{product_eval_univariate_accumulate};
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
        let mut sums: [F; D] = [F::ZERO; D];

        // Calculate the bitmask for the number of free variables
        let bitmask: usize = 1 << (self.num_free_variables() - 1);

        // First round: read from streams; subsequent rounds: read from tables
        if self.current_round == 0 {
            let streams = self.streams.as_ref().expect("streams present in round 0");
            let evaluations_len = 1usize << streams[0].num_variables();
            for i in 0..(evaluations_len / 2) {
                let mut pairs: [(F, F); D] = [(F::ZERO, F::ZERO); D];
                for j in 0..D {
                    let v0 = streams[j].evaluation(i);
                    let v1 = streams[j].evaluation(i | bitmask);
                    pairs[j] = (v0, v1);
                }
                product_eval_univariate_accumulate::<F, D>(&pairs, &mut sums);
            }
        } else {
            let tables: [&[F]; D] = std::array::from_fn(|j| {
                self.evaluations[j]
                    .as_ref()
                    .expect("tables present after round 0")
                    .as_slice()
            });
            let evaluations_len = tables[0].len();
            for i in 0..(evaluations_len / 2) {
                let mut pairs: [(F, F); D] = [(F::ZERO, F::ZERO); D];
                for j in 0..D {
                    let v0 = tables[j][i];
                    let v1 = tables[j][i | bitmask];
                    pairs[j] = (v0, v1);
                }
                product_eval_univariate_accumulate::<F, D>(&pairs, &mut sums);
            }
        }

        sums.to_vec()
    }
    pub fn reduce_evaluations(&mut self, verifier_message: F) {
        for i in 0..D {
            // Calculate constants for this reduction
            let setbit: usize = 1 << (self.num_free_variables() - 1);

            if let Some(mut prev) = self.evaluations[i].take() {
                // In-place reduction when we have previous evaluations
                let evaluations_len = prev.len() / 2;
                for i0 in 0..evaluations_len {
                    let i1 = i0 | setbit;
                    let e0 = prev[i0];
                    let e1 = prev[i1];
                    let diff = e1 - e0;
                    prev[i0] = e0 + diff * verifier_message;
                }
                prev.truncate(evaluations_len);
                self.evaluations[i] = Some(prev);
            } else {
                // First reduction: build from streams directly
                let evaluations_len = 1usize << self.num_free_variables();
                let streams = self
                    .streams
                    .as_ref()
                    .expect("Both streams and evaluations cannot be None");
                let mut out: Vec<F> = vec![F::ZERO; evaluations_len];
                for i0 in 0..evaluations_len {
                    let i1 = i0 | setbit;
                    let e0 = streams[i].evaluation(i0);
                    let e1 = streams[i].evaluation(i1);
                    let diff = e1 - e0;
                    out[i0] = e0 + diff * verifier_message;
                }
                self.evaluations[i] = Some(out);
            }
        }
    }
}
