#[cfg(test)]
use crate::{
    prover::Prover,
    streams::{multivariate_product_claim, MemoryStream, Stream},
    tests::{BenchStream, BN254},
    ProductSumcheck,
};
#[cfg(test)]
use crate::interpolation::multivariate::{compute_strides, multivariate_product_evaluations};
#[cfg(test)]
use crate::interpolation::univariate::product_eval_univariate_full;
#[cfg(test)]
use ark_ff::Zero;

/// Compare StreamingEvalProduct against Time (VSBW) baseline: identical round messages and lengths.
#[cfg(test)]
fn run_compare_equal_eval_product<const D: usize>(num_variables: usize) {
    use crate::multilinear_product::{StreamingEvalProductProver, StreamingEvalProductProverConfig, TimeProductProver, TimeProductProverConfig};

    // Build D identical streams from lex evaluations, then reorder to SignificantBit
    let s_tmp: BenchStream<BN254> = BenchStream::<BN254>::new(num_variables).into();
    let mut evals: Vec<BN254> = Vec::with_capacity(1 << num_variables);
    for i in 0..(1 << num_variables) { evals.push(s_tmp.evaluation(i)); }
    
    let base: MemoryStream<BN254> = MemoryStream::new_from_lex::<crate::order_strategy::SignificantBitOrder>(evals.clone()).into();
    let streams: Vec<MemoryStream<BN254>> = (0..D).map(|_| base.clone()).collect();
    let claim: BN254 = multivariate_product_claim(streams.clone());

    // Time (VSBW) baseline
    let time_transcript: ProductSumcheck<BN254, D> = ProductSumcheck::<BN254, D>::prove::<MemoryStream<BN254>, TimeProductProver<BN254, MemoryStream<BN254>, D>>(
        &mut Prover::<BN254>::new(TimeProductProverConfig { claim, num_variables, streams: streams.clone() }),
        &mut ark_std::test_rng(),
    );

    // Streaming evaluation candidate
    let eval_product_transcript: ProductSumcheck<BN254, D> = ProductSumcheck::<BN254, D>::prove::<MemoryStream<BN254>, StreamingEvalProductProver<BN254, MemoryStream<BN254>, D>>(
        &mut Prover::<BN254>::new(StreamingEvalProductProverConfig { claim, num_stages: 2, num_variables, streams }),
        &mut ark_std::test_rng(),
    );

    // Equal number of rounds and exact equality of nodes per round
    let r = time_transcript.prover_messages.len();
    assert_eq!(eval_product_transcript.prover_messages.len(), r);
    for k in 0..r { assert_eq!(time_transcript.prover_messages[k], eval_product_transcript.prover_messages[k]); }
}

#[test]
fn eval_product_equals_time_d2() { run_compare_equal_eval_product::<2>(6); }

#[test]
fn eval_product_equals_time_d3() { run_compare_equal_eval_product::<3>(6); }

#[test]
fn eval_product_equals_time_d8() { run_compare_equal_eval_product::<8>(6); }

#[test]
fn eval_product_equals_time_tiny_d8() { run_compare_equal_eval_product::<8>(3); }

#[test]
fn multivariate_product_v1_vs_univariate_full() {
	use crate::tests::BN254 as F;
	use ark_std::rand::{RngCore, SeedableRng};
	use ark_std::rand::rngs::StdRng;
	let mut rng = StdRng::seed_from_u64(1234567);
	for &d in &[2usize, 3, 4, 8] {
		let mut pairs: Vec<(F,F)> = Vec::with_capacity(d);
		let mut ml_inputs_01: Vec<Vec<F>> = Vec::with_capacity(d);
		for _ in 0..d {
			let a0 = F::from(rng.next_u64());
			let a1 = F::from(rng.next_u64());
			pairs.push((a0, a1));
			ml_inputs_01.push(vec![a0, a1]);
		}
		let mv = multivariate_product_evaluations::<F>(1, &ml_inputs_01, d);
		let uv = product_eval_univariate_full::<F>(&pairs);
		// Check ∞ equals product of slopes in both kernels
		let mut lc = F::from(1u64);
		for (a0, a1) in pairs.iter() { lc *= *a1 - *a0; }
		let uv_inf = *uv.last().unwrap();
		assert_eq!(uv_inf, lc, "univariate infinity not equal to product of slopes for d={}", d);
		assert_eq!(mv[d], lc, "mv infinity not equal to product of slopes for d={}", d);
		// Compare overlapping finite nodes: uv at nodes {1..D-1} vs mv at indices {0..D-2}
		for i in 1..d {
			assert_eq!(mv[i - 1], uv[i], "finite mismatch at node {} for d={}", i, d);
		}
	}
}

