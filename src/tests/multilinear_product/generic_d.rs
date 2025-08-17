use crate::{
    interpolation::LagrangePolynomial,
    order_strategy::GraycodeOrder,
    prover::Prover,
    streams::{multivariate_product_claim, MemoryStream, Stream},
    tests::{BenchStream, F64},
    ProductSumcheck,
};

#[allow(dead_code)]
fn run_d3_single<P>(num_variables: usize)
where
    P: Prover<F64, VerifierMessage = Option<F64>, ProverMessage = Option<Vec<F64>>>,
    P::ProverConfig: crate::prover::ProductProverConfig<F64, MemoryStream<F64>>,
{
    // Build three streams in SigBit order from lex evaluations
    let s_tmp: BenchStream<F64> = BenchStream::<F64>::new(num_variables).into();
    let mut evals: Vec<F64> = Vec::with_capacity(1 << num_variables);
    for i in 0..(1 << num_variables) {
        evals.push(s_tmp.evaluation(i));
    }
    let s1: MemoryStream<F64> = MemoryStream::new_from_lex::<crate::order_strategy::SignificantBitOrder>(evals.clone()).into();
    let s2 = s1.clone();
    let s3 = s1.clone();

    let claim: F64 = multivariate_product_claim(vec![s1.clone(), s2.clone(), s3.clone()]);

    // Prove
    let transcript: ProductSumcheck<F64> = ProductSumcheck::<F64>::prove::<MemoryStream<F64>, P>(
        &mut Prover::<F64>::new(<P::ProverConfig as crate::prover::ProductProverConfig<_, _>>::default(
            claim,
            num_variables,
            vec![s1, s2, s3],
        )),
        &mut ark_std::test_rng(),
    );

    assert!(transcript.is_accepted);

    // Stronger invariant checks: for each round k>=1,
    // t_{k-1}(r_{k-1}) == (t_k(0) + t_k(1))
    for k in 1..transcript.prover_messages.len() {
        let prev = &transcript.prover_messages[k - 1];
        let r = transcript.verifier_messages[k - 1];
        let eval = LagrangePolynomial::<F64, GraycodeOrder>::evaluate_from_points(r, prev);
        let round_sum = transcript.prover_messages[k][0] + transcript.prover_messages[k][1];
        assert_eq!(eval, round_sum);
    }
}

#[test]
fn d3_time_prover() {
    use crate::multilinear_product::TimeProductProver;
    run_d3_single::<TimeProductProver<F64, MemoryStream<F64>, 3>>(6);
}

#[test]
fn d3_space_prover() {
    use crate::multilinear_product::SpaceProductProver;
    run_d3_single::<SpaceProductProver<F64, MemoryStream<F64>, 3>>(6);
}

#[test]
fn d3_blendy_prover() {
    use crate::multilinear_product::BlendyProductProver;
    run_d3_single::<BlendyProductProver<F64, MemoryStream<F64>, 3>>(6);
}

#[cfg(test)]
fn run_compare_equal<const D: usize>(num_variables: usize) {
    use crate::multilinear_product::{BlendyProductProver, SpaceProductProver, TimeProductProver};

    // Build D streams
    let s_tmp: BenchStream<F64> = BenchStream::<F64>::new(num_variables).into();
    let mut evals: Vec<F64> = Vec::with_capacity(1 << num_variables);
    for i in 0..(1 << num_variables) { evals.push(s_tmp.evaluation(i)); }
    let base: MemoryStream<F64> = MemoryStream::new_from_lex::<crate::order_strategy::SignificantBitOrder>(evals.clone()).into();
    let streams: Vec<MemoryStream<F64>> = (0..D).map(|_| base.clone()).collect();
    let claim: F64 = multivariate_product_claim(streams.clone());

    // Time
    let time_transcript: ProductSumcheck<F64> = ProductSumcheck::<F64>::prove::<MemoryStream<F64>, TimeProductProver<F64, MemoryStream<F64>, D>>(
        &mut Prover::<F64>::new(crate::multilinear_product::TimeProductProverConfig { claim, num_variables, streams: streams.clone() }),
        &mut ark_std::test_rng(),
    );
    assert!(time_transcript.is_accepted);

    // Space
    let space_transcript: ProductSumcheck<F64> = ProductSumcheck::<F64>::prove::<MemoryStream<F64>, SpaceProductProver<F64, MemoryStream<F64>, D>>(
        &mut Prover::<F64>::new(crate::multilinear_product::SpaceProductProverConfig { claim, num_variables, streams: streams.clone() }),
        &mut ark_std::test_rng(),
    );
    assert!(space_transcript.is_accepted);

    // Blendy
    let blendy_transcript: ProductSumcheck<F64> = ProductSumcheck::<F64>::prove::<MemoryStream<F64>, BlendyProductProver<F64, MemoryStream<F64>, D>>(
        &mut Prover::<F64>::new(crate::multilinear_product::BlendyProductProverConfig { claim, num_stages: 2, num_variables, streams }),
        &mut ark_std::test_rng(),
    );
    assert!(blendy_transcript.is_accepted);

    // Equal number of rounds and exact equality of nodes per round
    let r = time_transcript.prover_messages.len();
    assert_eq!(space_transcript.prover_messages.len(), r);
    assert_eq!(blendy_transcript.prover_messages.len(), r);
    for k in 0..r {
        assert_eq!(time_transcript.prover_messages[k], space_transcript.prover_messages[k]);
        assert_eq!(time_transcript.prover_messages[k], blendy_transcript.prover_messages[k]);
    }
}

#[test]
fn compare_equal_d8() {
    run_compare_equal::<8>(6);
}

#[test]
fn compare_equal_d3() {
    run_compare_equal::<3>(6);
}

#[test]
fn compare_equal_d2() {
    run_compare_equal::<2>(6);
}


