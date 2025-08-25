use ark_ff::Field;
use ark_std::vec::Vec;

use crate::streams::Stream;
use crate::interpolation::univariate::product_eval_univariate_accumulate;
use crate::interpolation::field_mul_small::FieldMulSmall;

pub struct ImprovedTimeProductProver<F: Field, S: Stream<F>, const D: usize> {
    pub claim: F,
    pub current_round: usize,
    pub evaluations: Option<Vec<[F; D]>>, // AoS: per index, values for all D streams
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

        // Prefer tables if present (resilient when invoked mid-round), otherwise read from streams
        let (use_tables, evaluations_len) = match self.evaluations.as_ref() {
            Some(table) => (true, table.len()),
            None => {
                let streams = self
                    .streams
                    .as_ref()
                    .expect("Both streams and evaluations cannot be None");
                (false, 1usize << streams[0].num_variables())
            }
        };

        for i in 0..(evaluations_len / 2) {
            let mut pairs: [(F, F); D] = [(F::ZERO, F::ZERO); D];
            let i1 = i | bitmask;
            if use_tables {
                let table = self.evaluations.as_ref().unwrap();
                let row0 = table[i];
                let row1 = table[i1];
                for j in 0..D { pairs[j] = (row0[j], row1[j]); }
            } else {
                let streams = self.streams.as_ref().unwrap();
                for j in 0..D {
                    let v0 = streams[j].evaluation(i);
                    let v1 = streams[j].evaluation(i1);
                    pairs[j] = (v0, v1);
                }
            }
            product_eval_univariate_accumulate::<F, D>(&pairs, &mut sums);
        }

        sums.to_vec()
    }
    pub fn reduce_evaluations(&mut self, verifier_message: F) {
        // Calculate constants for this reduction
        let setbit: usize = 1 << self.num_free_variables();

        if let Some(mut prev) = self.evaluations.take() {
            // In-place reduction across all streams with AoS layout
            let evaluations_len = prev.len() / 2;
            for i0 in 0..evaluations_len {
                let i1 = i0 | setbit;
                let row0 = prev[i0];
                let row1 = prev[i1];
                let mut out_row: [F; D] = [F::ZERO; D];
                for j in 0..D {
                    let e0 = row0[j];
                    let e1 = row1[j];
                    let diff = e1 - e0;
                    out_row[j] = e0 + diff * verifier_message;
                }
                prev[i0] = out_row;
            }
            prev.truncate(evaluations_len);
            self.evaluations = Some(prev);
        } else {
            // First reduction: build from streams directly into AoS layout
            let evaluations_len = 1usize << self.num_free_variables();
            let streams = self
                .streams
                .as_ref()
                .expect("Both streams and evaluations cannot be None");
            let mut out: Vec<[F; D]> = vec![[F::ZERO; D]; evaluations_len];
            for i0 in 0..evaluations_len {
                let i1 = i0 | setbit;
                for j in 0..D {
                    let e0 = streams[j].evaluation(i0);
                    let e1 = streams[j].evaluation(i1);
                    let diff = e1 - e0;
                    out[i0][j] = e0 + diff * verifier_message;
                }
            }
            self.evaluations = Some(out);
        }
    }
}
