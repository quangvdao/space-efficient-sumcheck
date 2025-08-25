use ark_ff::Field;

use crate::{
    prover::{Prover, ProverConfig},
    streams::{MemoryStream, Stream},
    tests::polynomials::three_variable_polynomial_evaluations,
};

pub fn multilinear_round_sanity<F, S, P>(p: &mut P, message: Option<F>, eval_0: F, eval_1: F)
where
    F: Field,
    S: Stream<F>,
    P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F)>>,
{
    let round = p.next_message(message, F::ZERO).unwrap();
    assert_eq!(round.0, eval_0, "g0 should evaluate correctly",);
    assert_eq!(round.1, eval_1, "g1 should evaluate correctly",);
}

pub fn sanity_test<F, S, P>()
where
    F: Field,
    S: Stream<F> + From<MemoryStream<F>>,
    P: Prover<F, VerifierMessage = Option<F>, ProverMessage = Option<(F, F)>>,
    P::ProverConfig: ProverConfig<F, S>,
{
    let s: S = MemoryStream::new(three_variable_polynomial_evaluations()).into();
    let mut p = P::new(ProverConfig::default(F::from(6_u32), 3, s));
    /*
     * Zeroth Round: All variables are free
     *
     * Evaluations at different input points:
     *   (0,0,0) →  0
     *   (0,0,1) →  0
     *   (0,1,0) → 13
     *   (0,1,1) →  1
     *   -----------------
     *   Sum g₀(0) ≡ 14
     *
     *   (1,0,0) →  2
     *   (1,0,1) →  2
     *   (1,1,0) →  0
     *   (1,1,1) →  7
     *   -----------------
     *   Sum g₀(1) ≡ 11
     */
    multilinear_round_sanity::<F, S, P>(&mut p, None, F::from(14_u32), F::from(11_u32));
    /*
     * First Round: x₀ fixed to 3
     *
     * Evaluations at different input points:
     *   (3,0,1) →  6
     *   (3,0,0) →  6
     *   -----------------
     *   Sum g₁(0) ≡ 12
     *
     *   (3,1,1) → 38 ≡  0 (mod 19)
     *   (3,1,0) → 31 ≡ 12 (mod 19)
     *   -----------------
     *   Sum g₁(1) ≡ 12
     */
    multilinear_round_sanity::<F, S, P>(
        &mut p,
        Some(F::from(3_u32)),
        F::from(12_u32),
        F::from(12_u32),
    );
    /*
     * Last Round: x₁ fixed to 4
     *
     * Evaluations at different input points:
     *   (3,4,0) → 108 ≡ 11 (mod 19)
     *   -----------------
     *   Sum g(0) ≡ 11
     *
     *   (3,4,1) → 134 ≡  1 (mod 19)
     *   -----------------
     *   Sum g(1) ≡ 1
     */
    multilinear_round_sanity::<F, S, P>(
        &mut p,
        Some(F::from(4_u32)),
        F::from(11_u32),
        F::from(1_u32),
    );
}
