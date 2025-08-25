use ark_bn254::Fr as BN254Field;
use ark_ff::Field;

use space_efficient_sumcheck::{
    // hypercube::Hypercube,
    multilinear::{
        BlendyProver, BlendyProverConfig, SpaceProver, SpaceProverConfig, TimeProver,
        TimeProverConfig,
    },
    multilinear_product::{
        BlendyProductProver, BlendyProductProverConfig, ImprovedTimeProductProver,
        ImprovedTimeProductProverConfig, SpaceProductProver, SpaceProductProverConfig,
        TimeProductProver, TimeProductProverConfig,
    },
    // order_strategy::SignificantBitOrder,
    prover::{Prover, ProverConfig},
    streams::{multivariate_claim, multivariate_product_claim},
    tests::{BenchStream, F128, F64},
    ProductSumcheck,
    Sumcheck,
};

pub mod validation;
use validation::{validate_and_format_command_line_args, AlgorithmLabel, BenchArgs, FieldLabel};

fn run_on_field<F: Field>(bench_args: BenchArgs) {
    let mut rng = ark_std::test_rng();
    let s = BenchStream::<F>::new(bench_args.num_variables);

    // switch on algorithm_label
    match bench_args.algorithm_label {
        AlgorithmLabel::Blendy => {
            let config: BlendyProverConfig<F, BenchStream<F>> =
                BlendyProverConfig::<F, BenchStream<F>>::default(
                    multivariate_claim(s.clone()),
                    bench_args.num_variables,
                    s,
                );
            let _transcript = Sumcheck::<F>::prove::<BenchStream<F>, BlendyProver<F, BenchStream<F>>>(
                &mut BlendyProver::<F, BenchStream<F>>::new(config),
                &mut rng,
            );
        }
        AlgorithmLabel::VSBW => {
            let config: TimeProverConfig<F, BenchStream<F>> =
                TimeProverConfig::<F, BenchStream<F>>::default(
                    multivariate_claim(s.clone()),
                    bench_args.num_variables,
                    s,
                );
            let _transcript = Sumcheck::<F>::prove::<BenchStream<F>, TimeProver<F, BenchStream<F>>>(
                &mut TimeProver::<F, BenchStream<F>>::new(config),
                &mut rng,
            );
        }
        AlgorithmLabel::CTY => {
            let config: SpaceProverConfig<F, BenchStream<F>> =
                SpaceProverConfig::<F, BenchStream<F>>::default(
                    multivariate_claim(s.clone()),
                    bench_args.num_variables,
                    s,
                );
            let _transcript = Sumcheck::<F>::prove::<BenchStream<F>, SpaceProver<F, BenchStream<F>>>(
                &mut SpaceProver::<F, BenchStream<F>>::new(config),
                &mut rng,
            );
        }
        AlgorithmLabel::ProductVSBW => {
            let config: TimeProductProverConfig<F, BenchStream<F>> =
                TimeProductProverConfig::<F, BenchStream<F>> {
                    claim: multivariate_product_claim(vec![s.clone(); bench_args.d]),
                    num_variables: bench_args.num_variables,
                    streams: vec![s.clone(); bench_args.d],
                };
            match bench_args.d {
                2 => {
                    let _ = ProductSumcheck::<F, 2>::prove::<
                        BenchStream<F>,
                        TimeProductProver<F, BenchStream<F>, 2>,
                    >(
                        &mut TimeProductProver::<F, BenchStream<F>, 2>::new(config),
                        &mut rng,
                    );
                }
                3 => {
                    let _ = ProductSumcheck::<F, 3>::prove::<
                        BenchStream<F>,
                        TimeProductProver<F, BenchStream<F>, 3>,
                    >(
                        &mut TimeProductProver::<F, BenchStream<F>, 3>::new(config),
                        &mut rng,
                    );
                }
                4 => {
                    let _ = ProductSumcheck::<F, 4>::prove::<
                        BenchStream<F>,
                        TimeProductProver<F, BenchStream<F>, 4>,
                    >(
                        &mut TimeProductProver::<F, BenchStream<F>, 4>::new(config),
                        &mut rng,
                    );
                }
                8 => {
                    let _ = ProductSumcheck::<F, 8>::prove::<
                        BenchStream<F>,
                        TimeProductProver<F, BenchStream<F>, 8>,
                    >(
                        &mut TimeProductProver::<F, BenchStream<F>, 8>::new(config),
                        &mut rng,
                    );
                }
                16 => {
                    let _ = ProductSumcheck::<F, 16>::prove::<
                        BenchStream<F>,
                        TimeProductProver<F, BenchStream<F>, 16>,
                    >(
                        &mut TimeProductProver::<F, BenchStream<F>, 16>::new(config),
                        &mut rng,
                    );
                }
                32 => {
                    let _ = ProductSumcheck::<F, 32>::prove::<
                        BenchStream<F>,
                        TimeProductProver<F, BenchStream<F>, 32>,
                    >(
                        &mut TimeProductProver::<F, BenchStream<F>, 32>::new(config),
                        &mut rng,
                    );
                }
                _ => panic!("Unsupported d value: {}", bench_args.d),
            }
        }
        AlgorithmLabel::ProductBlendy => {
            let config: BlendyProductProverConfig<F, BenchStream<F>> =
                BlendyProductProverConfig::<F, BenchStream<F>> {
                    claim: multivariate_product_claim(vec![s.clone(); bench_args.d]),
                    num_variables: bench_args.num_variables,
                    num_stages: bench_args.stage_size,
                    streams: vec![s.clone(); bench_args.d],
                };
            match bench_args.d {
                2 => {
                    let _ = ProductSumcheck::<F, 2>::prove::<
                        BenchStream<F>,
                        BlendyProductProver<F, BenchStream<F>, 2>,
                    >(
                        &mut BlendyProductProver::<F, BenchStream<F>, 2>::new(config),
                        &mut rng,
                    );
                }
                3 => {
                    let _ = ProductSumcheck::<F, 3>::prove::<
                        BenchStream<F>,
                        BlendyProductProver<F, BenchStream<F>, 3>,
                    >(
                        &mut BlendyProductProver::<F, BenchStream<F>, 3>::new(config),
                        &mut rng,
                    );
                }
                4 => {
                    let _ = ProductSumcheck::<F, 4>::prove::<
                        BenchStream<F>,
                        BlendyProductProver<F, BenchStream<F>, 4>,
                    >(
                        &mut BlendyProductProver::<F, BenchStream<F>, 4>::new(config),
                        &mut rng,
                    );
                }
                8 => {
                    let _ = ProductSumcheck::<F, 8>::prove::<
                        BenchStream<F>,
                        BlendyProductProver<F, BenchStream<F>, 8>,
                    >(
                        &mut BlendyProductProver::<F, BenchStream<F>, 8>::new(config),
                        &mut rng,
                    );
                }
                16 => {
                    let _ = ProductSumcheck::<F, 16>::prove::<
                        BenchStream<F>,
                        BlendyProductProver<F, BenchStream<F>, 16>,
                    >(
                        &mut BlendyProductProver::<F, BenchStream<F>, 16>::new(config),
                        &mut rng,
                    );
                }
                32 => {
                    let _ = ProductSumcheck::<F, 32>::prove::<
                        BenchStream<F>,
                        BlendyProductProver<F, BenchStream<F>, 32>,
                    >(
                        &mut BlendyProductProver::<F, BenchStream<F>, 32>::new(config),
                        &mut rng,
                    );
                }
                _ => panic!("Unsupported d value: {}", bench_args.d),
            }
        }
        AlgorithmLabel::ProductCTY => {
            let config: SpaceProductProverConfig<F, BenchStream<F>> =
                SpaceProductProverConfig::<F, BenchStream<F>> {
                    claim: multivariate_product_claim(vec![s.clone(); bench_args.d]),
                    num_variables: bench_args.num_variables,
                    streams: vec![s.clone(); bench_args.d],
                };
            match bench_args.d {
                2 => {
                    let _ = ProductSumcheck::<F, 2>::prove::<
                        BenchStream<F>,
                        SpaceProductProver<F, BenchStream<F>, 2>,
                    >(
                        &mut SpaceProductProver::<F, BenchStream<F>, 2>::new(config),
                        &mut rng,
                    );
                }
                3 => {
                    let _ = ProductSumcheck::<F, 3>::prove::<
                        BenchStream<F>,
                        SpaceProductProver<F, BenchStream<F>, 3>,
                    >(
                        &mut SpaceProductProver::<F, BenchStream<F>, 3>::new(config),
                        &mut rng,
                    );
                }
                4 => {
                    let _ = ProductSumcheck::<F, 4>::prove::<
                        BenchStream<F>,
                        SpaceProductProver<F, BenchStream<F>, 4>,
                    >(
                        &mut SpaceProductProver::<F, BenchStream<F>, 4>::new(config),
                        &mut rng,
                    );
                }
                8 => {
                    let _ = ProductSumcheck::<F, 8>::prove::<
                        BenchStream<F>,
                        SpaceProductProver<F, BenchStream<F>, 8>,
                    >(
                        &mut SpaceProductProver::<F, BenchStream<F>, 8>::new(config),
                        &mut rng,
                    );
                }
                16 => {
                    let _ = ProductSumcheck::<F, 16>::prove::<
                        BenchStream<F>,
                        SpaceProductProver<F, BenchStream<F>, 16>,
                    >(
                        &mut SpaceProductProver::<F, BenchStream<F>, 16>::new(config),
                        &mut rng,
                    );
                }
                32 => {
                    let _ = ProductSumcheck::<F, 32>::prove::<
                        BenchStream<F>,
                        SpaceProductProver<F, BenchStream<F>, 32>,
                    >(
                        &mut SpaceProductProver::<F, BenchStream<F>, 32>::new(config),
                        &mut rng,
                    );
                }
                _ => panic!("Unsupported d value: {}", bench_args.d),
            }
        }
        AlgorithmLabel::ProductImprovedTime => unreachable!("routed in main to BN254-only runner"),
    };
}

fn run_product_improved_time<
    F: Field + space_efficient_sumcheck::interpolation::field_mul_small::FieldMulSmall,
>(
    bench_args: BenchArgs,
) {
    let mut rng = ark_std::test_rng();
    let s = BenchStream::<F>::new(bench_args.num_variables);
    let config: ImprovedTimeProductProverConfig<F, BenchStream<F>> =
        ImprovedTimeProductProverConfig::<F, BenchStream<F>> {
            claim: multivariate_product_claim(vec![s.clone(); bench_args.d]),
            num_variables: bench_args.num_variables,
            streams: vec![s.clone(); bench_args.d],
        };
    match bench_args.d {
        2 => {
            let _ = ProductSumcheck::<F, 2>::prove::<
                BenchStream<F>,
                ImprovedTimeProductProver<F, BenchStream<F>, 2>,
            >(
                &mut ImprovedTimeProductProver::<F, BenchStream<F>, 2>::new(config),
                &mut rng,
            );
        }
        3 => {
            let _ = ProductSumcheck::<F, 3>::prove::<
                BenchStream<F>,
                ImprovedTimeProductProver<F, BenchStream<F>, 3>,
            >(
                &mut ImprovedTimeProductProver::<F, BenchStream<F>, 3>::new(config),
                &mut rng,
            );
        }
        4 => {
            let _ = ProductSumcheck::<F, 4>::prove::<
                BenchStream<F>,
                ImprovedTimeProductProver<F, BenchStream<F>, 4>,
            >(
                &mut ImprovedTimeProductProver::<F, BenchStream<F>, 4>::new(config),
                &mut rng,
            );
        }
        8 => {
            let _ = ProductSumcheck::<F, 8>::prove::<
                BenchStream<F>,
                ImprovedTimeProductProver<F, BenchStream<F>, 8>,
            >(
                &mut ImprovedTimeProductProver::<F, BenchStream<F>, 8>::new(config),
                &mut rng,
            );
        }
        16 => {
            let _ = ProductSumcheck::<F, 16>::prove::<
                BenchStream<F>,
                ImprovedTimeProductProver<F, BenchStream<F>, 16>,
            >(
                &mut ImprovedTimeProductProver::<F, BenchStream<F>, 16>::new(config),
                &mut rng,
            );
        }
        32 => {
            let _ = ProductSumcheck::<F, 32>::prove::<
                BenchStream<F>,
                ImprovedTimeProductProver<F, BenchStream<F>, 32>,
            >(
                &mut ImprovedTimeProductProver::<F, BenchStream<F>, 32>::new(config),
                &mut rng,
            );
        }
        _ => panic!("Unsupported d value: {}", bench_args.d),
    }
}

fn main() {
    // Collect command line arguments
    let bench_args: BenchArgs = validate_and_format_command_line_args(std::env::args().collect());
    // Route ProductImprovedTime to BN254-only runner to satisfy small-mul bounds
    if let AlgorithmLabel::ProductImprovedTime = bench_args.algorithm_label {
        match bench_args.field_label {
            FieldLabel::FieldBn254 => run_product_improved_time::<BN254Field>(bench_args),
            _ => panic!("ProductImprovedTime currently supported only on BN254"),
        }
    } else {
        // Run the requested bench normally
        match bench_args.field_label {
            FieldLabel::Field64 => {
                run_on_field::<F64>(bench_args);
            }
            FieldLabel::Field128 => {
                run_on_field::<F128>(bench_args);
            }
            FieldLabel::FieldBn254 => {
                run_on_field::<BN254Field>(bench_args);
            }
        };
    }
}
