#[cfg(test)]
use crate::{
    prover::Prover,
    streams::{multivariate_product_claim, MemoryStream, Stream},
    tests::{BenchStream, BN254},
    ProductSumcheck,
};

#[cfg(test)]
fn run_compare_blendy_vs_baseline<const D: usize>(num_variables: usize) {
    use crate::multilinear_product::{BlendyProductProver, SpaceProductProver, TimeProductProver};

    // Build D streams
    let s_tmp: BenchStream<BN254> = BenchStream::<BN254>::new(num_variables).into();
    let mut evals: Vec<BN254> = Vec::with_capacity(1 << num_variables);
    for i in 0..(1 << num_variables) { 
        evals.push(s_tmp.evaluation(i)); 
    }
    let base: MemoryStream<BN254> = MemoryStream::new_from_lex::<crate::order_strategy::SignificantBitOrder>(evals.clone()).into();
    let streams: Vec<MemoryStream<BN254>> = (0..D).map(|_| base.clone()).collect();
    let claim: BN254 = multivariate_product_claim(streams.clone());

    // Time prover (baseline)
    let time_transcript: ProductSumcheck<BN254, D> = ProductSumcheck::<BN254, D>::prove::<MemoryStream<BN254>, TimeProductProver<BN254, MemoryStream<BN254>, D>>(
        &mut Prover::<BN254>::new(crate::multilinear_product::TimeProductProverConfig { 
            claim, 
            num_variables, 
            streams: streams.clone() 
        }),
        &mut ark_std::test_rng(),
    );

    // Space prover (baseline)
    let space_transcript: ProductSumcheck<BN254, D> = ProductSumcheck::<BN254, D>::prove::<MemoryStream<BN254>, SpaceProductProver<BN254, MemoryStream<BN254>, D>>(
        &mut Prover::<BN254>::new(crate::multilinear_product::SpaceProductProverConfig { 
            claim, 
            num_variables, 
            streams: streams.clone() 
        }),
        &mut ark_std::test_rng(),
    );

    // Blendy prover (with new scheduling)
    let blendy_transcript: ProductSumcheck<BN254, D> = ProductSumcheck::<BN254, D>::prove::<MemoryStream<BN254>, BlendyProductProver<BN254, MemoryStream<BN254>, D>>(
        &mut Prover::<BN254>::new(crate::multilinear_product::BlendyProductProverConfig { 
            claim, 
            num_stages: 2, 
            num_variables, 
            streams 
        }),
        &mut ark_std::test_rng(),
    );

    // Verify all have same number of rounds
    let r = time_transcript.prover_messages.len();
    assert_eq!(space_transcript.prover_messages.len(), r, "Space prover should have same rounds as time");
    assert_eq!(blendy_transcript.prover_messages.len(), r, "Blendy prover should have same rounds as time");

    // Verify exact equality of messages per round
    for k in 0..r {
        assert_eq!(time_transcript.prover_messages[k], space_transcript.prover_messages[k], 
                   "Time and space should match at round {}", k);
        assert_eq!(time_transcript.prover_messages[k], blendy_transcript.prover_messages[k], 
                   "Time and blendy should match at round {}", k);
    }
    
    println!("✓ Blendy correctness verified for D={}, num_variables={}", D, num_variables);
}

#[test]
fn blendy_correctness_d2() { 
    run_compare_blendy_vs_baseline::<2>(6); 
}

#[test]
fn blendy_correctness_d3() { 
    run_compare_blendy_vs_baseline::<3>(6); 
}

#[test]
fn blendy_correctness_d4() { 
    run_compare_blendy_vs_baseline::<4>(6); 
}

#[test]
fn blendy_correctness_d8() { 
    run_compare_blendy_vs_baseline::<8>(6); 
}
