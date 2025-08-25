use ark_ff::Field;

use crate::{
    multilinear_product::provers::time_with_eq::TimeEqProverConfig,
    prover::Prover,
    streams::Stream,
};
use crate::multilinear_product::provers::time_with_eq::time_with_eq::TimeProductProverWithEq;

impl<F: Field, S: Stream<F>, const D: usize> Prover<F> for TimeProductProverWithEq<F, S, D> {
    type ProverConfig = TimeEqProverConfig<F, S>;
    type ProverMessage = Option<Vec<F>>;
    type VerifierMessage = Option<F>;

    fn claim(&self) -> F { self.claim }

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
            evaluations: std::array::from_fn(|_| None),
            streams: Some(streams_arr),
            num_variables,
            w: prover_config.w,
            prefix_alpha: F::ONE,
        }
    }

    fn next_message(&mut self, verifier_message: Option<F>, claim_sum: F) -> Option<Vec<F>> {
        if self.current_round >= self.total_rounds() { return None; }
        self.claim = claim_sum; // Update claim before evaluate
        if self.current_round != 0 {
            self.reduce_evaluations(verifier_message.unwrap());
        }
        let sums = self.evaluate();
        self.current_round += 1;
        Some(sums)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        multilinear_product::{TimeProductProver, TimeProductProverConfig},
        tests::{BenchStream, F64},
    };
    use crate::prover::Prover;
    use crate::streams::multivariate_product_claim;

    #[test]
    fn eq_constant_matches_scaled_baseline_all_rounds() {
        const D: usize = 2;
        let n = 6;
        let s = BenchStream::<F64>::new(n);
        let claim = multivariate_product_claim(vec![s.clone(); D]);
        let mut base = TimeProductProver::<F64, BenchStream<F64>, D>::new(TimeProductProverConfig::new(
            claim,
            n,
            vec![s.clone(); D],
        ));
        let two = F64::from(2u64);
        let half = two.inverse().unwrap();
        let w = vec![half; n];
        let mut eqp = TimeProductProverWithEq::<F64, BenchStream<F64>, D>::new(TimeEqProverConfig::new(
            claim,
            n,
            vec![s.clone(); D],
            w,
        ));
        let c = F64::ONE; // overall eq constant multiplier accumulates as (1/2)^n, but independent of round
        for _ in 0..n {
            // Compare messages this round
            let sums_base = base.vsbw_evaluate();
            let sums_eq = eqp.evaluate();
            for (a, b) in sums_base.iter().zip(sums_eq.iter()) {
                assert_eq!(*b, *a * c);
            }
            // Sample a fixed r and advance both
            let r = F64::from(7u64); // arbitrary non-0, non-1
            base.vsbw_reduce_evaluations(r);
            eqp.reduce_evaluations(r);
        }
    }
}
