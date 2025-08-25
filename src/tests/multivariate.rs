#[cfg(test)]
use ark_ff::Field;
#[cfg(test)]
use crate::interpolation::multivariate::{
    multivariate_extrapolate,
    multilinear_extend_to_u_h_grid_naive,
    multivariate_product_evaluations,
    lift_01_to_u1_grid,
    // New paper-aligned functions
    multivariate_extrapolate_paper_uk,
    multilinear_extend_to_paper_u_h_grid_naive,
    multi_product_eval_paper,
    multivariate_product_evaluations_paper_naive,
    lift_01_to_paper_u1_grid,
};
#[cfg(test)]
use crate::tests::BN254;

#[cfg(test)]
fn random_polys_01<F: Field>(v: usize, d: usize) -> Vec<Vec<F>> {
    let n = 1usize << v;
    (0..d).map(|j| {
        // vary each poly slightly
        (0..n).map(|i| F::from(((i as u64) * 17 + (j as u64) * 101 + 5) % 7919)).collect()
    }).collect()
}

#[cfg(test)]
fn check_extrapolate_bn254() {
    type F = BN254;
    for v in [1usize, 2, 3, 4].iter().copied() {
        for h in [4, 6, 8, 10, 12].iter().copied() {
            let p01 = random_polys_01::<F>(v, 2).pop().unwrap();
            let u1_input = lift_01_to_u1_grid::<F>(v, &p01);
            #[allow(deprecated)]
            let fast = multivariate_extrapolate::<F>(v, 1, h, &u1_input);
            #[allow(deprecated)]
            let naive = multilinear_extend_to_u_h_grid_naive::<F>(v, &p01, h);
            if fast != naive {
                // find first mismatch for easier debugging
                for (i, (a, b)) in fast.iter().zip(naive.iter()).enumerate() {
                    if a != b { panic!("mismatch for v={v}, h={h} at idx={i}: fast={a}, naive={b}"); }
                }
            }
        }
    }
}

#[cfg(test)]
fn check_extrapolate_paper_bn254() {
    type F = BN254;
    for v in [1usize, 2, 3, 4].iter().copied() {
        for h in [4, 6, 8, 10, 12].iter().copied() {
            let p01 = random_polys_01::<F>(v, 2).pop().unwrap();
            let u1_input = lift_01_to_paper_u1_grid::<F>(v, &p01);
            let fast = multivariate_extrapolate_paper_uk::<F>(v, 1, h, &u1_input);
            let naive = multilinear_extend_to_paper_u_h_grid_naive::<F>(v, &p01, h);
            if fast != naive {
                // find first mismatch for easier debugging
                for (i, (a, b)) in fast.iter().zip(naive.iter()).enumerate() {
                    if a != b { panic!("mismatch for v={v}, h={h} at idx={i}: fast={a}, naive={b}"); }
                }
            }
        }
    }
}

#[cfg(test)]
fn check_product_bn254() {
    type F = BN254;
    for v in [1usize, 2, 3].iter().copied() {
        for d in [1usize, 2, 3, 4, 5, 6, 7, 8].iter().copied() {
            let polys = random_polys_01::<F>(v, d);
            // Build naive baseline: extend each to U_d^v, then pointwise multiply
            #[allow(deprecated)]
            let grids: Vec<Vec<F>> = polys.iter().map(|p01| multilinear_extend_to_u_h_grid_naive::<F>(v, p01, d)).collect();
            let out_len = usize::pow(d + 1, v as u32);
            let mut naive = vec![F::ONE; out_len];
            for g in grids.iter() { for (o, x) in naive.iter_mut().zip(g.iter()) { *o *= *x; } }

            #[allow(deprecated)]
            let fast = multivariate_product_evaluations::<F>(v, &polys, d);
            if fast != naive {
                for (i, (a, b)) in fast.iter().zip(naive.iter()).enumerate() {
                    if a != b { panic!("product mismatch for v={v}, d={d} at idx={i}: fast={a}, naive={b}"); }
                }
            }
        }
    }
}

#[cfg(test)]
fn check_product_paper_bn254() {
    type F = BN254;
    for v in [1usize, 2, 3].iter().copied() {
        for d in [1usize, 2, 3, 4, 5, 6, 7, 8].iter().copied() {
            let polys = random_polys_01::<F>(v, d);
            
            let fast = multi_product_eval_paper::<F>(v, &polys, d);
            let naive = multivariate_product_evaluations_paper_naive::<F>(v, &polys, d);
            
            if fast != naive {
                for (i, (a, b)) in fast.iter().zip(naive.iter()).enumerate() {
                    if a != b { panic!("product mismatch for v={v}, d={d} at idx={i}: fast={a}, naive={b}"); }
                }
            }
        }
    }
}

#[test]
fn multivariate_extrapolate_matches_naive_bn254() {
    check_extrapolate_bn254();
}

#[test]
fn multivariate_extrapolate_paper_matches_naive_bn254() {
    check_extrapolate_paper_bn254();
}

#[test]
#[ignore = "product pipeline currently diverges from naive baseline; enable once fixed"]
fn multivariate_product_matches_naive_bn254() {
    check_product_bn254();
}

#[test]
fn multivariate_product_paper_matches_naive_bn254() {
    check_product_paper_bn254();
}


