use crate::{
    prover::Prover,
    streams::{multivariate_product_claim, MemoryStream, Stream},
    tests::{BenchStream, BN254},
    ProductSumcheck,
};

#[cfg(test)]
fn debug_eval_product_vs_time<const D: usize>(num_variables: usize, num_stages: usize) {
    use crate::multilinear_product::{StreamingEvalProductProver, TimeProductProver, StreamingEvalProductProverConfig, TimeProductProverConfig};

    // Build D streams
    let s_tmp: BenchStream<BN254> = BenchStream::<BN254>::new(num_variables).into();
    let mut evals: Vec<BN254> = Vec::with_capacity(1 << num_variables);
    for i in 0..(1 << num_variables) { 
        evals.push(s_tmp.evaluation(i)); 
    }
    let base: MemoryStream<BN254> = MemoryStream::new_from_lex::<crate::order_strategy::SignificantBitOrder>(evals.clone()).into();
    let streams: Vec<MemoryStream<BN254>> = (0..D).map(|_| base.clone()).collect();
    let claim: BN254 = multivariate_product_claim(streams.clone());

    println!("=== DEBUG: D={}, n={}, k={}, claim={:?} ===", D, num_variables, num_stages, claim);

    // Time prover (baseline)
    let time_transcript: ProductSumcheck<BN254, D> = ProductSumcheck::<BN254, D>::prove::<MemoryStream<BN254>, TimeProductProver<BN254, MemoryStream<BN254>, D>>(
        &mut Prover::<BN254>::new(TimeProductProverConfig { 
            claim, 
            num_variables, 
            streams: streams.clone() 
        }),
        &mut ark_std::test_rng(),
    );
    
    // Eval product prover
    let eval_transcript: ProductSumcheck<BN254, D> = ProductSumcheck::<BN254, D>::prove::<MemoryStream<BN254>, StreamingEvalProductProver<BN254, MemoryStream<BN254>, D>>(
        &mut Prover::<BN254>::new(StreamingEvalProductProverConfig { 
            claim, 
            num_stages, 
            num_variables, 
            streams 
        }),
        &mut ark_std::test_rng(),
    );

    // Compare transcripts
    println!("Time transcript rounds: {}", time_transcript.prover_messages.len());
    println!("Eval transcript rounds: {}", eval_transcript.prover_messages.len());
    
    if time_transcript.prover_messages.len() != eval_transcript.prover_messages.len() {
        panic!("Different number of rounds! Time: {}, Eval: {}", 
               time_transcript.prover_messages.len(), eval_transcript.prover_messages.len());
    }
    
    for (round, (time_poly, eval_poly)) in time_transcript.prover_messages.iter().zip(eval_transcript.prover_messages.iter()).enumerate() {
        println!("\n--- Round {} ---", round);
        println!("Time poly: {:?}", time_poly);
        println!("Eval poly: {:?}", eval_poly);
        
        if time_poly != eval_poly {
            println!("❌ MISMATCH at round {}!", round);
            println!("Time coeffs: {:?}", time_poly);
            println!("Eval coeffs: {:?}", eval_poly);
            panic!("Eval product gives different result than time prover!");
        }
    }
    
    println!("✅ All rounds match!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_eval_product_d2_small() {
        debug_eval_product_vs_time::<2>(6, 2);
    }
    
    #[test] 
    fn debug_eval_product_d3_small() {
        debug_eval_product_vs_time::<3>(6, 2);
    }
}
