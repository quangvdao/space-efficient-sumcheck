use ark_bn254::Fr as Fp;
use ark_ff::AdditiveGroup;
use ark_std::time::Instant;

use space_efficient_sumcheck::interpolation::{
    univariate::{product_eval_univariate_accumulate, product_eval_univariate_accumulate_from_v1_diff},
};
use space_efficient_sumcheck::multilinear_product::{
    ImprovedTimeProductProver, ImprovedTimeProductProverConfig,
    TimeProductProver, TimeProductProverConfig,
    TimeProductProverWithEq, TimeEqProverConfig,
};
use space_efficient_sumcheck::streams::multivariate_product_claim;
use space_efficient_sumcheck::tests::BenchStream;
use space_efficient_sumcheck::prover::Prover;

fn bench_one<const D: usize>(iters: usize) {
    // random-ish deterministic inputs
    let mut pairs: [(Fp, Fp); D] = [(Fp::ZERO, Fp::ZERO); D];
    let mut seed = Fp::from(7u64);
    for j in 0..D {
        seed += seed;
        let a0 = seed;
        let a1 = a0 + Fp::from(3u64);
        pairs[j] = (a0, a1);
    }

    let mut sums: [Fp; D] = [Fp::ZERO; D];
    let start = Instant::now();
    for _ in 0..iters {
        // zero sums
        for s in sums.iter_mut() { *s = Fp::ZERO; }
        product_eval_univariate_accumulate::<Fp, D>(&pairs, &mut sums);
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_secs_f64() * 1e9;
    println!("D={}, iters={}, total_ns={:.0}, per_call_ns={:.1}", D, iters, ns, ns / iters as f64);

    // A/B: v1,diff path
    let mut pairs_v1_diff: [(Fp, Fp); D] = [(Fp::ZERO, Fp::ZERO); D];
    for j in 0..D { pairs_v1_diff[j] = (pairs[j].1, pairs[j].1 - pairs[j].0); }
    let mut sums2: [Fp; D] = [Fp::ZERO; D];
    let start2 = Instant::now();
    for _ in 0..iters {
        for s in sums2.iter_mut() { *s = Fp::ZERO; }
        product_eval_univariate_accumulate_from_v1_diff::<Fp, D>(&pairs_v1_diff, &mut sums2);
    }
    let elapsed2 = start2.elapsed();
    let ns2 = elapsed2.as_secs_f64() * 1e9;
    println!("D={} (v1,diff), iters={}, total_ns={:.0}, per_call_ns={:.1}", D, iters, ns2, ns2 / iters as f64);
}

fn main() {
    // Light iterations to keep runtime short
    bench_one::<4>(200_000);
    bench_one::<8>(200_000);
    bench_one::<16>(100_000);
    bench_one::<32>(50_000);

    // Prover-level eval vs reduce microbench (one round worth), D in {4,8,16,32}
    fn bench_prover<const D: usize>(n_vars: usize) {
        let s = BenchStream::<Fp>::new(n_vars);
        // ImprovedTime eval
        let cfg_it: ImprovedTimeProductProverConfig<Fp, BenchStream<Fp>> = ImprovedTimeProductProverConfig {
            claim: multivariate_product_claim(vec![s.clone(); D]),
            num_variables: n_vars,
            streams: vec![s.clone(); D],
        };
        let prover_it = &ImprovedTimeProductProver::<Fp, BenchStream<Fp>, D>::new(cfg_it);
        let t0 = Instant::now();
        let _ = prover_it.toom_evaluate();
        let dt_eval = t0.elapsed();

        // ImprovedTime reduce (first reduction, from streams)
        let mut prover_it_red = ImprovedTimeProductProver::<Fp, BenchStream<Fp>, D>::new(ImprovedTimeProductProverConfig {
            claim: multivariate_product_claim(vec![s.clone(); D]),
            num_variables: n_vars,
            streams: vec![s.clone(); D],
        });
        prover_it_red.current_round = 1; // mimic next round
        let r = Fp::from(3u64);
        let t1 = Instant::now();
        prover_it_red.reduce_evaluations(r);
        let dt_reduce = t1.elapsed();

        // VSBW eval
        let cfg_vs: TimeProductProverConfig<Fp, BenchStream<Fp>> = TimeProductProverConfig {
            claim: multivariate_product_claim(vec![s.clone(); D]),
            num_variables: n_vars,
            streams: vec![s.clone(); D],
        };
        let prover_vs = &TimeProductProver::<Fp, BenchStream<Fp>, D>::new(cfg_vs);
        let t2 = Instant::now();
        let _ = prover_vs.vsbw_evaluate();
        let dt_vs_eval = t2.elapsed();

        // VSBW reduce
        let mut prover_vs_red = TimeProductProver::<Fp, BenchStream<Fp>, D>::new(TimeProductProverConfig {
            claim: multivariate_product_claim(vec![s.clone(); D]),
            num_variables: n_vars,
            streams: vec![s.clone(); D],
        });
        prover_vs_red.current_round = 1;
        let t3 = Instant::now();
        prover_vs_red.vsbw_reduce_evaluations(r);
        let dt_vs_reduce = t3.elapsed();

        // Time-with-eq eval (k_parts = 2, w = geometric pattern)
        let mut w: Vec<Fp> = vec![Fp::ZERO; n_vars];
        let mut t = Fp::from(5u64);
        for i in 0..n_vars { t += t; w[i] = t; }
        let cfg_eq: TimeEqProverConfig<Fp, BenchStream<Fp>> = TimeEqProverConfig::new(
            multivariate_product_claim(vec![s.clone(); D]), n_vars, vec![s.clone(); D], w,
        );
        let prover_eq = &TimeProductProverWithEq::<Fp, BenchStream<Fp>, D>::new(cfg_eq);
        let t4 = Instant::now();
        let _ = prover_eq.evaluate();
        let dt_eq_eval = t4.elapsed();

        println!(
            "Prover D={} eval_ns={} reduce_ns={} vsbw_eval_ns={} vsbw_reduce_ns={} eq_eval_ns={}",
            D,
            (dt_eval.as_secs_f64()*1e9) as u64,
            (dt_reduce.as_secs_f64()*1e9) as u64,
            (dt_vs_eval.as_secs_f64()*1e9) as u64,
            (dt_vs_reduce.as_secs_f64()*1e9) as u64,
            (dt_eq_eval.as_secs_f64()*1e9) as u64,
        );
    }

    bench_prover::<4>(20);
    bench_prover::<8>(20);
    bench_prover::<16>(20);
    bench_prover::<32>(20);

    // Memory-scaling eval microbench: vary slice count to probe memory ops
    fn bench_eval_memory<const D: usize>(p: usize) {
        // num_slices = 2^p; arrays length = 2*num_slices; bitmask = num_slices
        let num_slices: usize = 1 << p;
        let arr_len = num_slices * 2;
        let setbit = num_slices;
        // Build synthetic arrays per factor
        let mut a0: Vec<Vec<Fp>> = Vec::with_capacity(D);
        let mut a1: Vec<Vec<Fp>> = Vec::with_capacity(D);
        let mut seed = Fp::from(5u64);
        for _ in 0..D {
            let mut v0 = vec![Fp::ZERO; arr_len];
            let mut v1 = vec![Fp::ZERO; arr_len];
            for i in 0..arr_len {
                seed += seed;
                v0[i] = seed;
                v1[i] = v0[i] + Fp::from(7u64);
            }
            a0.push(v0);
            a1.push(v1);
        }
        let mut sums: [Fp; D] = [Fp::ZERO; D];
        let start = Instant::now();
        for i in 0..num_slices {
            // gather pairs
            let mut pairs: [(Fp, Fp); D] = [(Fp::ZERO, Fp::ZERO); D];
            for j in 0..D {
                let v0 = a0[j][i];
                let v1 = a1[j][i | setbit];
                pairs[j] = (v0, v1);
            }
            product_eval_univariate_accumulate::<Fp, D>(&pairs, &mut sums);
        }
        let ns = (start.elapsed().as_secs_f64() * 1e9) as u64;
        println!(
            "MemEval D={} slices={} total_ns={} per_slice_ns={}",
            D,
            num_slices,
            ns,
            ns / (num_slices as u64)
        );
    }

    // Run for moderate sizes to limit memory
    bench_eval_memory::<16>(8);   // 256 slices
    bench_eval_memory::<16>(12);  // 4096 slices
    bench_eval_memory::<16>(15);  // 32768 slices
    bench_eval_memory::<32>(8);
    bench_eval_memory::<32>(12);
    bench_eval_memory::<32>(15);
}


