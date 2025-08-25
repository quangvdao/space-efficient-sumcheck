#[cfg(test)]
use crate::{
    prover::Prover,
    streams::{multivariate_product_claim, MemoryStream, Stream},
    tests::{BenchStream, BN254},
    ProductSumcheck,
};

#[cfg(test)]
mod test_improved_time {
    use ark_bn254::Fr;
    use ark_ff::{UniformRand, AdditiveGroup};
    use ark_std::vec::Vec;
    use ark_std::rand::{SeedableRng, rngs::StdRng};

    use crate::interpolation::univariate::{
        product_eval_univariate_accumulate, product_eval_univariate_full,
    };

    fn run_test_product_eval_accumulate<const D: usize>() {
        let mut rng = StdRng::seed_from_u64(42);
        let pairs: [(Fr, Fr); D] = (0..D)
            .map(|_| (Fr::rand(&mut rng), Fr::rand(&mut rng)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Ground truth from `product_eval_univariate_full`
        let full_evals = product_eval_univariate_full(&pairs);
        let mut expected = Vec::with_capacity(D);
        if D > 1 {
            // g(1), g(2), ..., g(D-1)
            for i in 1..D {
                expected.push(full_evals[i]);
            }
            // g(inf) is the last element
            expected.push(*full_evals.last().unwrap());
        } else {
            // D=1 case is special, message is just g(1)
            // `full_evals` for D=1 returns [g(0), g(inf)], but g(1) is just p_1(1)
            expected.push(pairs[0].1);
        }

        // Result from `product_eval_univariate_accumulate`
        let mut sums = vec![Fr::ZERO; if D > 1 { D } else { 1 }];
        product_eval_univariate_accumulate::<Fr, D>(&pairs, &mut sums);

        assert_eq!(expected, sums, "Mismatch for D={}", D);
    }

    #[test]
    fn test_product_eval_accumulate_d2() {
        run_test_product_eval_accumulate::<2>();
    }

    #[test]
    fn test_product_eval_accumulate_d4() {
        run_test_product_eval_accumulate::<4>();
    }

    #[test]
    fn test_product_eval_accumulate_d8() {
        run_test_product_eval_accumulate::<8>();
    }

    #[test]
    fn test_product_eval_accumulate_d16() {
        run_test_product_eval_accumulate::<16>();
    }

    #[test]
    fn test_product_eval_accumulate_d32() {
        run_test_product_eval_accumulate::<32>();
    }
}

/// Compare ImprovedTime against Time (VSBW) baseline: identical round messages.
#[cfg(test)]
fn run_compare_equal_improved_time<const D: usize>(num_variables: usize) {
    use crate::multilinear_product::{ImprovedTimeProductProver, ImprovedTimeProductProverConfig, TimeProductProver, TimeProductProverConfig};

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

    // ImprovedTime candidate
    let improved_time_transcript: ProductSumcheck<BN254, D> = ProductSumcheck::<BN254, D>::prove::<MemoryStream<BN254>, ImprovedTimeProductProver<BN254, MemoryStream<BN254>, D>>(
        &mut Prover::<BN254>::new(ImprovedTimeProductProverConfig { claim, num_variables, streams }),
        &mut ark_std::test_rng(),
    );

    // Equal number of rounds and exact equality of nodes per round
    let r = time_transcript.prover_messages.len();
    assert_eq!(improved_time_transcript.prover_messages.len(), r);
    for k in 0..r { assert_eq!(time_transcript.prover_messages[k], improved_time_transcript.prover_messages[k]); }
}

#[test]
fn improved_time_equals_time_d2() { run_compare_equal_improved_time::<2>(6); }

#[test]
fn improved_time_equals_time_d3() { run_compare_equal_improved_time::<3>(6); }

#[test]
fn improved_time_equals_time_d8() { run_compare_equal_improved_time::<8>(6); }


