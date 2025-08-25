use ark_ff::Field;
use ark_std::vec::Vec;
use super::field_mul_small::FieldMulSmall;

// Small helpers mirroring high-d-opt
#[inline]
fn dbl<F: FieldMulSmall>(x: F) -> F { <F as FieldMulSmall>::double(&x) }

#[inline]
fn mul6<F: FieldMulSmall>(x: F) -> F { x.mul_u64(6) }

#[inline]
fn pow_usize<F: Field>(base: F, mut exp: usize) -> F {
    let mut acc = F::ONE;
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= b;
        }
        b *= b;
        exp >>= 1;
    }
    acc
}

// -----------------------------------------------------------------------------
// Improved univariate extrapolation/evaluation (ported from high-d-opt)
// -----------------------------------------------------------------------------

// d = 2: [0, 1] -> [1, 2, inf]
#[inline]
fn eval_inter2<F: FieldMulSmall>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F, F) {
    let p_inf = p1 - p0;
    let p2 = p_inf + p1;
    let q_inf = q1 - q0;
    let q2 = q_inf + q1;
    let r1 = p1 * q1;
    let r2 = p2 * q2;
    let r_inf = p_inf * q_inf;
    (r1, r2, r_inf)
}

// d = 2: [0, 1] -> [1, inf]
#[inline]
fn eval_inter2_final<F: FieldMulSmall>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F) {
    let r1 = p1 * q1;
    let r_inf = (p1 - p0) * (q1 - q0);
    (r1, r_inf)
}

// d = 4: [1, 2, inf] -> [1, 2, 3, 4, inf]
#[inline]
fn eval_inter4<F: FieldMulSmall>(p: [(F, F); 4]) -> (F, F, F, F, F) {
    #[inline]
    fn helper<F: FieldMulSmall>(fx0: F, fx1: F, f_inf: F) -> F {
        dbl(fx1 + f_inf) - fx0
    }
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = helper(a1, a2, a_inf);
    let a4 = helper(a2, a3, a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = helper(b1, b2, b_inf);
    let b4 = helper(b2, b3, b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a4 * b4, a_inf * b_inf)
}

// d = 4: [1, 2, inf] -> [1, 2, 3, inf]
#[inline]
fn eval_inter4_final<F: FieldMulSmall>(p: [(F, F); 4]) -> (F, F, F, F) {
    #[inline]
    fn helper<F: FieldMulSmall>(fx0: F, fx1: F, f_inf: F) -> F {
        dbl(fx1 + f_inf) - fx0
    }
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = helper(a1, a2, a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = helper(b1, b2, b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a_inf * b_inf)
}

// d = 3: [1, 2, inf] (for A = p0*p1) with linear p2 -> [1, 2, 3, inf]
#[inline]
fn eval_inter3<F: FieldMulSmall>(p: [(F, F); 3]) -> [F; 4] {
    #[inline]
    fn helper<F: FieldMulSmall>(fx0: F, fx1: F, f_inf: F) -> F {
        // same forward-step as in eval_inter4: f_{t+1} = 2*(f_t + f_inf) - f_{t-1}
        dbl(fx1 + f_inf) - fx0
    }
    // First combine p0 and p1 to get quadratic A on {1,2,inf}, then step to 3
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = helper(a1, a2, a_inf);
    // Third linear factor values at 1,2,3 and inf
    let (c0, c1) = p[2];
    let c_inf = c1 - c0;
    let c2 = c1 + c_inf; // 2*c1 - c0
    let c3 = c2 + c_inf; // 3*c1 - 2*c0
    [
        a1 * c1,
        a2 * c2,
        a3 * c3,
        a_inf * c_inf,
    ]
}

// d = 8: [1, 2, 3, 4, inf] -> [1, 2, 3, 4, 5, 6, 7, 8, inf]
#[inline]
fn eval_inter8<F: FieldMulSmall>(p: [(F, F); 8]) -> [F; 9] {
    #[inline]
    fn helper_pair<F: FieldMulSmall>(f: &[F; 4], f_inf6: F) -> (F, F) {
        let f3m2 = f[3] - f[2];
        let f4 = dbl(dbl(f_inf6 + f3m2 + f[1]) - f[2]) - f[0];
        let f5 = dbl(dbl(f4 - f3m2 + f_inf6) - f[3]) - f[1];
        (f4, f5)
    }
    #[inline]
    fn batch_helper<F: FieldMulSmall>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F, F) {
        let f_inf6 = mul6(f_inf);
        let (f4, f5) = helper_pair(&[f0, f1, f2, f3], f_inf6);
        let (f6, f7) = helper_pair(&[f2, f3, f4, f5], f_inf6);
        (f4, f5, f6, f7)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(p[0..4].try_into().unwrap());
    let (a5, a6, a7, a8) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(p[4..8].try_into().unwrap());
    let (b5, b6, b7, b8) = batch_helper(b1, b2, b3, b4, b_inf);
    [
        a1 * b1,
        a2 * b2,
        a3 * b3,
        a4 * b4,
        a5 * b5,
        a6 * b6,
        a7 * b7,
        a8 * b8,
        a_inf * b_inf,
    ]
}

// d = 8: [1, 2, 3, 4, inf] -> [1, 2, 3, 4, 5, 6, 7, 8, inf]
#[inline]
fn eval_inter8_final<F: FieldMulSmall>(p: [(F, F); 8]) -> [F; 8] {
    #[inline]
    fn helper_pair<F: FieldMulSmall>(f: &[F; 4], f_inf6: F) -> (F, F) {
        let f3m2 = f[3] - f[2];
        let f4 = dbl(dbl(f_inf6 + f3m2 + f[1]) - f[2]) - f[0];
        let f5 = dbl(dbl(f4 - f3m2 + f_inf6) - f[3]) - f[1];
        (f4, f5)
    }
    #[inline]
    fn batch_helper<F: FieldMulSmall>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = mul6(f_inf);
        let (f4, f5) = helper_pair(&[f0, f1, f2, f3], f_inf6);
        let f6 = dbl(dbl(f_inf6 + f5 - f4 + f3) - f4) - f2;
        (f4, f5, f6)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(p[0..4].try_into().unwrap());
    let (a5, a6, a7) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(p[4..8].try_into().unwrap());
    let (b5, b6, b7) = batch_helper(b1, b2, b3, b4, b_inf);
    [
        a1 * b1,
        a2 * b2,
        a3 * b3,
        a4 * b4,
        a5 * b5,
        a6 * b6,
        a7 * b7,
        a_inf * b_inf,
    ]
}

// Fused accumulate variants for layout [1, 2, ..., D-1, ∞]
#[inline(always)]
fn eval_inter2_final_accumulate<F: FieldMulSmall>(p0: (F, F), p1: (F, F), sums: &mut [F]) {
    let r1 = p0.1 * p1.1; // g(1)
    let r_inf = (p0.1 - p0.0) * (p1.1 - p1.0); // ∞
    sums[0] += r1;
    sums[1] += r_inf;
}

#[inline(always)]
fn eval_inter4_final_accumulate<F: FieldMulSmall>(p: &[(F, F); 4], sums: &mut [F]) {
    #[inline]
    fn helper<F: FieldMulSmall>(fx0: F, fx1: F, f_inf: F) -> F {
        dbl(fx1 + f_inf) - fx0
    }
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = helper(a1, a2, a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = helper(b1, b2, b_inf);
    sums[0] += a1 * b1; // 1
    sums[1] += a2 * b2; // 2
    sums[2] += a3 * b3; // 3
    sums[3] += a_inf * b_inf; // ∞
}

#[inline(always)]
fn eval_inter8_final_accumulate<F: FieldMulSmall>(p: &[(F, F); 8], sums: &mut [F]) {
    // A direct port of the logic from eval_inter8_final, but accumulating.
    #[inline]
    fn helper_pair<F: FieldMulSmall>(f: &[F; 4], f_inf6: F) -> (F, F) {
        let f3m2 = f[3] - f[2];
        let f4 = dbl(dbl(f_inf6 + f3m2 + f[1]) - f[2]) - f[0];
        let f5 = dbl(dbl(f4 - f3m2 + f_inf6) - f[3]) - f[1];
        (f4, f5)
    }
    #[inline]
    fn batch_helper<F: FieldMulSmall>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = mul6(f_inf);
        let (f4, f5) = helper_pair(&[f0, f1, f2, f3], f_inf6);
        let f6 = dbl(dbl(f_inf6 + f5 - f4 + f3) - f4) - f2;
        (f4, f5, f6)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(p[0..4].try_into().unwrap());
    let (a5, a6, a7) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(p[4..8].try_into().unwrap());
    let (b5, b6, b7) = batch_helper(b1, b2, b3, b4, b_inf);

    sums[0] += a1 * b1;
    sums[1] += a2 * b2;
    sums[2] += a3 * b3;
    sums[3] += a4 * b4;
    sums[4] += a5 * b5;
    sums[5] += a6 * b6;
    sums[6] += a7 * b7;
    sums[7] += a_inf * b_inf;
}

// d = 16: [1, 2, ..., 7, 8, inf] -> [1, 2, ..., 16, inf]
#[inline]
fn eval_inter16<F: FieldMulSmall>(p: [(F, F); 16]) -> [F; 17] {
    #[inline]
    fn helper_idx<F: FieldMulSmall>(f: &[F; 17], b: usize, f_inf40320: F) -> F {
        // window is f[b-8..b]
        let a1 = f[b - 7] + f[b - 1]; // f[1]+f[7]
        let a2 = f[b - 5] + f[b - 3]; // f[3]+f[5]
        let n1 = f[b - 6] + f[b - 2]; // f[2]+f[6]
        let n2 = f[b - 4];           // f[4]
        let n3 = f[b - 8];           // f[0]
        F::linear_combination_i64(&[(a1, 8), (a2, 56), (f_inf40320, 1)], &[(n1, 28), (n2, 70), (n3, 1)])
    }
    #[inline]
    fn batch_helper<F: FieldMulSmall>(vals: &[F; 9]) -> [F; 17] {
        let mut f = [F::ZERO; 17]; // f[1, ..., 16, inf]
        for i in 0..8 { f[i] = vals[i]; }
        f[16] = vals[8];
        let f_inf40320 = vals[8].mul_u64(40320); // 8!
        for i in 8..16 { f[i] = helper_idx(&f, i, f_inf40320); }
        f
    }
    let a = batch_helper(&eval_inter8(p[0..8].try_into().unwrap()));
    let b = batch_helper(&eval_inter8(p[8..16].try_into().unwrap()));
    let mut res = [F::ZERO; 17];
    for i in 0..17 { res[i] = a[i] * b[i]; }
    res
}

// d = 16: [1, 2, ..., 7, 8, inf] -> [1, 2, ..., 15, inf]
#[inline]
fn eval_inter16_final<F: FieldMulSmall>(p: [(F, F); 16]) -> [F; 16] {
    #[inline]
    fn helper_idx<F: FieldMulSmall>(f: &[F; 16], b: usize, f_inf40320: F) -> F {
        // window is f[b-8..b]
        let a1 = f[b - 7] + f[b - 1]; // f[1]+f[7]
        let a2 = f[b - 5] + f[b - 3]; // f[3]+f[5]
        let n1 = f[b - 6] + f[b - 2]; // f[2]+f[6]
        let n2 = f[b - 4];           // f[4]
        let n3 = f[b - 8];           // f[0]
        F::linear_combination_i64(&[(a1, 8), (a2, 56), (f_inf40320, 1)], &[(n1, 28), (n2, 70), (n3, 1)])
    }
    #[inline]
    fn batch_helper<F: FieldMulSmall>(vals: &[F; 9]) -> [F; 16] {
        let mut f = [F::ZERO; 16]; // f[1, ..., 15, inf]
        for i in 0..8 { f[i] = vals[i]; }
        f[15] = vals[8];
        let f_inf40320 = vals[8].mul_u64(40320); // 8!
        for i in 8..15 { f[i] = helper_idx(&f, i, f_inf40320); }
        f
    }
    let a = batch_helper(&eval_inter8(p[0..8].try_into().unwrap()));
    let b = batch_helper(&eval_inter8(p[8..16].try_into().unwrap()));
    let mut res = [F::ZERO; 16];
    for i in 0..16 { res[i] = a[i] * b[i]; }
    res
}

#[inline(always)]
fn eval_inter16_final_accumulate<F: FieldMulSmall>(p: &[(F, F); 16], sums: &mut [F]) {
    #[inline]
    fn helper_idx<F: FieldMulSmall>(f: &[F; 16], b: usize, f_inf40320: F) -> F {
        // window is f[b-8..b]
        let a1 = f[b - 7] + f[b - 1]; // f[1]+f[7]
        let a2 = f[b - 5] + f[b - 3]; // f[3]+f[5]
        let n1 = f[b - 6] + f[b - 2]; // f[2]+f[6]
        let n2 = f[b - 4];           // f[4]
        let n3 = f[b - 8];           // f[0]
        F::linear_combination_i64(&[(a1, 8), (a2, 56), (f_inf40320, 1)], &[(n1, 28), (n2, 70), (n3, 1)])
    }
    #[inline]
    fn batch_helper<F: FieldMulSmall>(vals: &[F; 9], sums: &mut [F], off: usize) {
        let mut f = [F::ZERO; 16]; // f[1..15, inf]
        for i in 0..8 { f[i] = vals[i]; }
        f[15] = vals[8];
        let f_inf40320 = vals[8].mul_u64(40320);
        for i in 8..15 { f[i] = helper_idx(&f, i, f_inf40320); }
        for i in 0..15 { sums[off + i] += f[i]; }
    }
    let a = eval_inter8(p[0..8].try_into().unwrap());
    let b = eval_inter8(p[8..16].try_into().unwrap());
    batch_helper(&a, sums, 0);
    batch_helper(&b, sums, 0);
}

// d = 32: [1, 2, ..., 16, inf] -> [1, 2, ..., 32, inf]
#[inline]
#[allow(dead_code)]
fn eval_inter32<F: FieldMulSmall>(p: [(F, F); 32]) -> [F; 33] {
    #[inline]
    fn helper<F: FieldMulSmall>(f: &[F; 16], f_infbig: F) -> F {
        F::linear_combination_i64(
            &[
                (f[1] + f[15], 16),
                (f[3] + f[13], 560),
                (f[5] + f[11], 4368),
                (f[7] + f[9], 11440),
                (f_infbig, 1),
            ],
            &[
                (f[2] + f[14], 120),
                (f[4] + f[12], 1820),
                (f[6] + f[10], 8008),
                (f[8], 12870),
                (f[0], 1),
            ],
        )
    }
    #[inline]
    fn batch_helper<F: FieldMulSmall>(vals: &[F; 17]) -> [F; 33] {
        let mut f = [F::ZERO; 33]; // f[1, ..., 32, inf]
        for i in 0..16 { f[i] = vals[i]; }
        f[32] = vals[16];
        let f_infbig = vals[16].mul_u64(20922789888000u64); // 16!
        for i in 16..32 { f[i] = helper(&f[(i - 16)..i].try_into().unwrap(), f_infbig); }
        f
    }
    let a = batch_helper(&eval_inter16(p[0..16].try_into().unwrap()));
    let b = batch_helper(&eval_inter16(p[16..32].try_into().unwrap()));
    let mut res = [F::ZERO; 33];
    for i in 0..33 { res[i] = a[i] * b[i]; }
    res
}

// d = 32: [1, 2, ..., 16, inf] -> [1, 2, ..., 31, inf]
#[inline]
fn eval_inter32_final<F: FieldMulSmall>(p: [(F, F); 32]) -> [F; 32] {
    #[inline]
    fn helper<F: FieldMulSmall>(f: &[F; 16], f_infbig: F) -> F {
        F::linear_combination_i64(
            &[
                (f[1] + f[15], 16),
                (f[3] + f[13], 560),
                (f[5] + f[11], 4368),
                (f[7] + f[9], 11440),
                (f_infbig, 1),
            ],
            &[
                (f[2] + f[14], 120),
                (f[4] + f[12], 1820),
                (f[6] + f[10], 8008),
                (f[8], 12870),
                (f[0], 1),
            ],
        )
    }
    #[inline]
    fn batch_helper<F: FieldMulSmall>(vals: &[F; 17]) -> [F; 32] {
        let mut f = [F::ZERO; 32]; // f[1, ..., 31, inf]
        for i in 0..16 { f[i] = vals[i]; }
        f[31] = vals[16];
        let f_infbig = vals[16].mul_u64(20922789888000u64); // 16!
        for i in 16..31 { f[i] = helper(&f[(i - 16)..i].try_into().unwrap(), f_infbig); }
        f
    }
    let a = batch_helper(&eval_inter16(p[0..16].try_into().unwrap()));
    let b = batch_helper(&eval_inter16(p[16..32].try_into().unwrap()));
    let mut res = [F::ZERO; 32];
    for i in 0..32 { res[i] = a[i] * b[i]; }
    res
}

#[inline(always)]
fn eval_inter32_final_accumulate<F: FieldMulSmall>(p: &[(F, F); 32], sums: &mut [F]) {
    #[inline]
    fn helper<F: FieldMulSmall>(f: &[F; 16], f_infbig: F) -> F {
        F::linear_combination_i64(
            &[
                (f[1] + f[15], 16),
                (f[3] + f[13], 560),
                (f[5] + f[11], 4368),
                (f[7] + f[9], 11440),
                (f_infbig, 1),
            ],
            &[
                (f[2] + f[14], 120),
                (f[4] + f[12], 1820),
                (f[6] + f[10], 8008),
                (f[8], 12870),
                (f[0], 1),
            ],
        )
    }
    #[inline]
    fn batch_helper<F: FieldMulSmall>(vals: &[F; 17], sums: &mut [F], off: usize) {
        let mut f = [F::ZERO; 32]; // f[1..31, inf]
        for i in 0..16 { f[i] = vals[i]; }
        f[31] = vals[16];
        let f_infbig = vals[16].mul_u64(20922789888000u64);
        for i in 16..31 { f[i] = helper(&f[(i - 16)..i].try_into().unwrap(), f_infbig); }
        for i in 0..31 { sums[off + i] += f[i]; }
    }
    let a = eval_inter16(p[0..16].try_into().unwrap());
    let b = eval_inter16(p[16..32].try_into().unwrap());
    batch_helper(&a, sums, 0);
    batch_helper(&b, sums, 0);
}

// -----------------------------------------------------------------------------
// Canonical univariate extrapolation: U_k = [1..k, ∞] → U_h = [1..h, ∞]
// Doubling steps use the same small-int recurrences as eval_inter{8,16,32} helpers.
// Tail is handled by extending to next power-of-two and truncating (keeps f(∞)).
// -----------------------------------------------------------------------------

#[inline]
fn extend1_to_2<F: FieldMulSmall>(vals: &[F; 2]) -> [F; 3] {
    let f1 = vals[0];
    let f_inf = vals[1];
    [f1, f1 + f_inf, f_inf]
}

#[inline]
fn extend2_to_4<F: FieldMulSmall>(vals: &[F; 3]) -> [F; 5] {
    let f1 = vals[0];
    let f2 = vals[1];
    let f_inf = vals[2];
    let f3 = -f1 + dbl(f2 + f_inf);                // -1*f1 + 2*f2 + 2*f_inf
    let f4 = -(dbl(f1)) + (f2 + f2 + f2) + f_inf.mul_u64(6); // -2*f1 + 3*f2 + 6*f_inf
    [f1, f2, f3, f4, f_inf]
}

#[inline]
fn extend4_to_8<F: FieldMulSmall>(vals: &[F; 5]) -> [F; 9] {
    #[inline]
    fn helper_pair<F: FieldMulSmall>(f: &[F; 4], f_inf6: F) -> (F, F) {
        let f3m2 = f[3] - f[2];
        let f4 = dbl(dbl(f_inf6 + f3m2 + f[1]) - f[2]) - f[0];
        let f5 = dbl(dbl(f4 - f3m2 + f_inf6) - f[3]) - f[1];
        (f4, f5)
    }
    let f1 = vals[0]; let f2 = vals[1]; let f3 = vals[2]; let f4 = vals[3]; let f_inf = vals[4];
    let f_inf6 = mul6(f_inf);
    let (f5, f6) = helper_pair(&[f1, f2, f3, f4], f_inf6);
    let (f7, f8) = helper_pair(&[f3, f4, f5, f6], f_inf6);
    [f1, f2, f3, f4, f5, f6, f7, f8, f_inf]
}

#[inline]
fn extend8_to_16<F: FieldMulSmall>(vals: &[F; 9]) -> [F; 17] {
    #[inline]
    fn helper<F: FieldMulSmall>(f: &[F; 8], f_inf40320: F) -> F {
        F::linear_combination_i64(&[(f[1] + f[7], 8), (f[3] + f[5], 56), (f_inf40320, 1)], &[
            (f[2] + f[6], 28),
            (f[4], 70),
            (f[0], 1),
        ])
    }
    let mut f = [F::ZERO; 17];
    for i in 0..8 { f[i] = vals[i]; }
    f[16] = vals[8];
    let f_inf40320 = vals[8].mul_u64(40320);
    for i in 8..16 { f[i] = helper(&f[(i - 8)..i].try_into().unwrap(), f_inf40320); }
    f
}

#[inline]
fn extend16_to_32<F: FieldMulSmall>(vals: &[F; 17]) -> [F; 33] {
    #[inline]
    fn helper<F: FieldMulSmall>(f: &[F; 16], f_infbig: F) -> F {
        F::linear_combination_i64(
            &[
                (f[1] + f[15], 16),
                (f[3] + f[13], 560),
                (f[5] + f[11], 4368),
                (f[7] + f[9], 11440),
                (f_infbig, 1),
            ],
            &[
                (f[2] + f[14], 120),
                (f[4] + f[12], 1820),
                (f[6] + f[10], 8008),
                (f[8], 12870),
                (f[0], 1),
            ],
        )
    }
    let mut f = [F::ZERO; 33];
    for i in 0..16 { f[i] = vals[i]; }
    f[32] = vals[16];
    let f_infbig = vals[16].mul_u64(20922789888000u64);
    for i in 16..32 { f[i] = helper(&f[(i - 16)..i].try_into().unwrap(), f_infbig); }
    f
}

pub fn extrapolate_uk_to_uh<F: FieldMulSmall>(values_uk: &[F], h: usize) -> Vec<F> {
    let mut k = values_uk.len() - 1;
    assert!(k >= 1 && k <= 16, "supported k in [1,2,4,8,16]");
    assert!(h >= k && h <= 32, "supported h up to 32");
    // Clone to a fixed-size array per step
    let mut cur: Vec<F> = values_uk.to_vec();
    while k < h {
        match k {
            1 => {
                let arr: [F; 2] = [cur[0], cur[1]];
                let ext = extend1_to_2(&arr);
                cur = ext.to_vec();
                k = 2;
            }
            2 => {
                let arr: [F; 3] = [cur[0], cur[1], cur[2]];
                let ext = extend2_to_4(&arr);
                cur = ext.to_vec();
                k = 4;
            }
            4 => {
                let arr: [F; 5] = [cur[0], cur[1], cur[2], cur[3], cur[4]];
                let ext = extend4_to_8(&arr);
                cur = ext.to_vec();
                k = 8;
            }
            8 => {
                let arr: [F; 9] = cur[0..9].try_into().unwrap();
                let ext = extend8_to_16(&arr);
                cur = ext.to_vec();
                k = 16;
            }
            16 => {
                let arr: [F; 17] = cur[0..17].try_into().unwrap();
                let ext = extend16_to_32(&arr);
                cur = ext.to_vec();
                k = 32;
            }
            _ => unreachable!(),
        }
    }
    if k == h { return cur; }
    // Truncate to h: keep f(1..h) and preserve the leading coefficient.
    let mut out = vec![F::ZERO; h + 1];
    for i in 0..h { out[i] = cur[i]; }
    out[h] = *cur.last().unwrap();
    out
}

#[inline]
fn eval_poly<F: Field>(coeffs: &[F], x: F) -> F {
    // Horner: coeffs[0] + coeffs[1] x + ... + coeffs[k] x^k
    let mut acc = F::ZERO;
    for c in coeffs.iter().rev() {
        acc *= x;
        acc += *c;
    }
    acc
}

// Solve V c = y where V_{i,j} = (i)^j for i,j = 0..k-1.
fn solve_vandermonde<F: Field>(y: &mut [F]) -> Vec<F> {
    let k = y.len();
    if k == 0 {
        return Vec::new();
    }
    // Augment V | y and perform Gaussian elimination. We build V on the fly.
    // Matrix a of size k x k
    let mut a: Vec<F> = vec![F::ZERO; k * k];
    for i in 0..k {
        let xi = F::from(i as u64);
        // fill row i with powers xi^0 .. xi^{k-1}
        let mut p = F::ONE;
        for j in 0..k {
            a[i * k + j] = p;
            p *= xi;
        }
    }
    // Forward elimination
    for col in 0..k {
        // find pivot
        let mut pivot = col;
        while pivot < k && a[pivot * k + col].is_zero() {
            pivot += 1;
        }
        assert!(pivot < k, "singular Vandermonde (duplicate nodes)");
        if pivot != col {
            // swap rows pivot <-> col without overlapping borrows
            for j in col..k {
                let idx_p = pivot * k + j;
                let idx_c = col * k + j;
                let tmp = a[idx_p];
                a[idx_p] = a[idx_c];
                a[idx_c] = tmp;
            }
            y.swap(pivot, col);
        }
        // normalize row
        let inv = a[col * k + col].inverse().expect("nonzero pivot");
        for j in col..k {
            a[col * k + j] *= inv;
        }
        y[col] *= inv;
        // eliminate below using a cached copy of the pivot row to avoid aliasing
        let pivot_row: Vec<F> = (col..k).map(|j| a[col * k + j]).collect();
        for row in (col + 1)..k {
            let factor = a[row * k + col];
            if factor.is_zero() { continue; }
            for (off, pval) in pivot_row.iter().enumerate() {
                let j = col + off;
                let idx = row * k + j;
                let prod = pval.clone() * factor;
                a[idx] -= prod;
            }
            y[row] -= factor * y[col];
        }
    }
    // Back substitution
    let mut c = vec![F::ZERO; k];
    for i in (0..k).rev() {
        let mut sum = y[i];
        for j in (i + 1)..k {
            sum -= a[i * k + j] * c[j];
        }
        c[i] = sum; // since diagonal is 1 after normalization
    }
    c
}

// Extrapolate from U_k (values at 0..k-1 plus leading coeff at ∞) to U_n (0..n-1, ∞).
fn extrapolate_from_u_k_to_u_n<F: Field>(values_u_k: &[F], n: usize) -> Vec<F> {
    let k = values_u_k.len() - 1; // degree k
    let leading = values_u_k[k];
    if k == 0 {
        // constant polynomial
        let mut out = vec![values_u_k[0]; n + 1];
        out[n] = F::ZERO; // leading coeff is 0 for degree 0
        return out;
    }
    // Build RHS y = values[0..k] - leading * i^k
    let mut y: Vec<F> = Vec::with_capacity(k);
    for i in 0..k {
        let xi = F::from(i as u64);
        let term = pow_usize(xi, k);
        y.push(values_u_k[i] - leading * term);
    }
    // Solve for coefficients c0..c_{k-1}
    let mut y_slice = y; // mutable copy for elimination
    let mut coeffs = solve_vandermonde::<F>(&mut y_slice);
    // Append c_k = leading
    coeffs.push(leading);
    // Evaluate on 0..n-1 and append ∞
    let mut out = vec![F::ZERO; n + 1];
    for x in 0..n {
        let xv = F::from(x as u64);
        out[x] = eval_poly(&coeffs, xv);
    }
    out[n] = leading;
    out
}

/// Compute the full table of the product of D linear polynomials on U_D,
/// where pairs[j] = (p_j(0), p_j(1)).
/// Returns [g(0), g(1), g(2), ..., g(D-1), g(∞)].
pub fn product_eval_univariate_full<F: FieldMulSmall>(pairs: &[(F, F)]) -> Vec<F> {
    let d = pairs.len();
    assert!(d >= 1, "need at least one linear factor");
    // Fast paths: powers of two call hand-optimized eval_inter*_final routines
    if d == 1 {
        let a0 = pairs[0].0;
        let slope = pairs[0].1 - a0;
        return vec![a0, slope];
    }
    if d == 2 {
        let arr: [(F, F); 2] = pairs.try_into().unwrap();
        let (r1, r_inf) = eval_inter2_final(arr[0], arr[1]); // [1, ∞]
        let g0 = arr[0].0 * arr[1].0;
        return vec![g0, r1, r_inf]; // [0, 1, ∞]
    }
    if d == 4 {
        let arr: [(F, F); 4] = pairs.try_into().unwrap();
        let (r1, r2, r3, r_inf) = eval_inter4_final(arr); // [1,2,3, ∞]
        let mut g0 = F::ONE;
        for (a0, _) in arr.iter().copied() { g0 *= a0; }
        let mut out = vec![F::ZERO; 5];
        out[0] = g0;
        out[1] = r1; out[2] = r2; out[3] = r3; out[4] = r_inf;
        return out;
    }
    if d == 3 {
        let arr: [(F, F); 3] = pairs.try_into().unwrap();
        let r = eval_inter3(arr); // [1,2,3, ∞]
        let mut g0 = F::ONE;
        for (a0, _) in arr.iter().copied() { g0 *= a0; }
        let mut out = vec![F::ZERO; 5];
        out[0] = g0;
        out[1] = r[0]; out[2] = r[1]; out[3] = r[2]; out[4] = r[3];
        return out;
    }
    if d == 8 {
        let arr: [(F, F); 8] = pairs.try_into().unwrap();
        let r = eval_inter8_final(arr); // [1..7, ∞]
        let mut g0 = F::ONE;
        for (a0, _) in arr.iter().copied() { g0 *= a0; }
        let mut out = vec![F::ZERO; 9];
        out[0] = g0;
        for i in 0..7 { out[1 + i] = r[i]; }
        out[8] = r[7];
        return out;
    }
    if d == 16 {
        let arr: [(F, F); 16] = pairs.try_into().unwrap();
        let r = eval_inter16_final(arr); // [1..15, ∞]
        let mut g0 = F::ONE;
        for (a0, _) in arr.iter().copied() { g0 *= a0; }
        let mut out = vec![F::ZERO; 17];
        out[0] = g0;
        for i in 0..15 { out[1 + i] = r[i]; }
        out[16] = r[15];
        return out;
    }
    if d == 32 {
        let arr: [(F, F); 32] = pairs.try_into().unwrap();
        let r = eval_inter32_final(arr); // [1..31, ∞]
        let mut g0 = F::ONE;
        for (a0, _) in arr.iter().copied() { g0 *= a0; }
        let mut out = vec![F::ZERO; 33];
        out[0] = g0;
        for i in 0..31 { out[1 + i] = r[i]; }
        out[32] = r[31];
        return out;
    }

    // Fallback (non power-of-two): divide & conquer with template-based extrapolation where possible
    fn rec<F: Field>(pairs: &[(F, F)]) -> Vec<F> {
        let k = pairs.len();
        if k == 1 {
            let a0 = pairs[0].0;
            let slope = pairs[0].1 - a0;
            let mut out = vec![F::ZERO; 1 + 1];
            out[0] = a0;
            out[1] = slope;
            return out;
        }
        let mid = k / 2;
        let left = rec::<F>(&pairs[..mid]);
        let right = rec::<F>(&pairs[mid..]);
        // Extrapolate to U_k using fallback method (powers-of-two are already handled above)
        let left_ext = extrapolate_from_u_k_to_u_n::<F>(&left, k);
        let right_ext = extrapolate_from_u_k_to_u_n::<F>(&right, k);
        let mut out = vec![F::ZERO; k + 1];
        for i in 0..=k { out[i] = left_ext[i] * right_ext[i]; }
        out
    }

    rec::<F>(pairs)
}

/// Fused variant: directly accumulates the contribution of the product table on U_D
/// into `sums` with message layout [1, 2, ..., D-1, ∞].
/// For D = 1, the message is [1].
/// Precondition: sums.len() == if D>1 { D } else { 1 }.
#[inline(always)]
pub fn product_eval_univariate_accumulate<F: FieldMulSmall, const D: usize>(pairs: &[(F, F); D], sums: &mut [F]) {
    debug_assert_eq!(sums.len(), if D > 1 { D } else { 1 });

    if D == 1 {
        let (_a0, a1) = pairs[0];
        sums[0] += a1; // g(1)
        return;
    }

    // Fast paths for small D using evaluation-based Toom-style formulas
    if D == 2 {
        // [1, ∞]
        eval_inter2_final_accumulate(pairs[0], pairs[1], sums);
        return;
    }
    if D == 3 {
        let (a0, a1) = pairs[0];
        let (b0, b1) = pairs[1];
        // quadratic A on {0,1,∞}
        let a0q = a0 * b0;
        let a1q = a1 * b1;
        let a_infq = (a1 - a0) * (b1 - b0);
        // extend A to -1
        let a0_2 = a0q + a0q;
        let a_inf_2 = a_infq + a_infq;
        let a_m1 = a0_2 - a1q + a_inf_2;
        // third linear
        let (c0, c1) = pairs[2];
        let c_inf = c1 - c0;
        let c_m1 = c0 - c_inf; // 2*c0 - c1
        // point-wise products
        let _r0 = a0q * c0;
        let r1 = a1q * c1; // g(1)
        let r_m1 = a_m1 * c_m1;
        let r_inf = a_infq * c_inf;
        // derive r2 from {0,1,-1,∞}
        let r2 = -_r0.mul_u64(3) + r1.mul_u64(3) + r_m1 + r_inf.mul_u64(6);
        // [1, 2, ∞]
        sums[0] += r1;
        sums[1] += r2;
        sums[2] += r_inf;
        return;
    }
    if D == 4 {
        // [1, 2, 3, ∞]
        let arr: &[(F, F); 4] = unsafe { &*(pairs as *const _ as *const [(F, F); 4]) };
        eval_inter4_final_accumulate(arr, sums);
        return;
    }

    // Fast paths for larger power-of-two D using improved univariate evaluation
    if D == 8 {
        // [1..7, ∞]
        let arr: &[(F, F); 8] = unsafe { &*(pairs as *const _ as *const [(F, F); 8]) };
        eval_inter8_final_accumulate(arr, sums);
        return;
    }
    if D == 16 {
        let arr: &[(F, F); 16] = unsafe { &*(pairs as *const _ as *const [(F, F); 16]) };
        eval_inter16_final_accumulate(arr, sums);
        return;
    }
    if D == 32 {
        let arr: &[(F, F); 32] = unsafe { &*(pairs as *const _ as *const [(F, F); 32]) };
        eval_inter32_final_accumulate(arr, sums);
        return;
    }

    // Recursive helper returning values on U_k for the subproblem size k.
    fn rec<F: Field>(pairs: &[(F, F)]) -> Vec<F> {
        let k = pairs.len();
        if k == 1 {
            let a0 = pairs[0].0;
            let slope = pairs[0].1 - a0;
            let mut out = vec![F::ZERO; 1 + 1];
            out[0] = a0; // x=0
            out[1] = slope; // ∞
            return out;
        }
        let mid = k / 2;
        let left = rec::<F>(&pairs[..mid]);
        let right = rec::<F>(&pairs[mid..]);
        let left_ext = extrapolate_from_u_k_to_u_n::<F>(&left, k);
        let right_ext = extrapolate_from_u_k_to_u_n::<F>(&right, k);
        let mut out = vec![F::ZERO; k + 1];
        for i in 0..=k { out[i] = left_ext[i] * right_ext[i]; }
        out
    }

    // Compute subproducts and extrapolate to U_D, then accumulate without materializing `out`.
    let mid = D / 2;
    let left = rec::<F>(&pairs[..mid]);
    let right = rec::<F>(&pairs[mid..]);
    let left_ext = extrapolate_from_u_k_to_u_n::<F>(&left, D);
    let right_ext = extrapolate_from_u_k_to_u_n::<F>(&right, D);

    // Accumulate message entries: [1, 2, ..., D-1, ∞]
    for k in 1..D { sums[k - 1] += left_ext[k] * right_ext[k]; }
    sums[D - 1] += left_ext[D] * right_ext[D];
}

/// Variant that accepts (v1, diff) pairs, where diff = v1 - v0 for each linear factor.
/// Reconstructs (v0, v1) once per factor, then delegates to the main kernel.
#[inline(always)]
pub fn product_eval_univariate_accumulate_from_v1_diff<F: FieldMulSmall, const D: usize>(pairs_v1_diff: &[(F, F)], sums: &mut [F]) {
    debug_assert_eq!(pairs_v1_diff.len(), D);
    if D == 1 {
        sums[0] += pairs_v1_diff[0].0; // g(1) = v1
        return;
    }
    if D == 2 {
        // [1, ∞]: g(1) = v1_a * v1_b; g(∞) = diff_a * diff_b
        let (a1, ainf) = (pairs_v1_diff[0].0, pairs_v1_diff[0].1);
        let (b1, binf) = (pairs_v1_diff[1].0, pairs_v1_diff[1].1);
        sums[0] += a1 * b1;
        sums[1] += ainf * binf;
        return;
    }
    if D == 4 {
        // Two halves of two factors each
        let (a1, ad) = (pairs_v1_diff[0].0, pairs_v1_diff[0].1);
        let (b1, bd) = (pairs_v1_diff[1].0, pairs_v1_diff[1].1);
        let (c1, cd) = (pairs_v1_diff[2].0, pairs_v1_diff[2].1);
        let (d1, dd) = (pairs_v1_diff[3].0, pairs_v1_diff[3].1);
        // Left pair values at 1,2,3 and ∞
        let la1 = a1 * b1;
        let la2 = (a1 + ad) * (b1 + bd);
        let la3 = (a1 + ad + ad) * (b1 + bd + bd);
        let lainf = ad * bd;
        // Right pair
        let rb1 = c1 * d1;
        let rb2 = (c1 + cd) * (d1 + dd);
        let rb3 = (c1 + cd + cd) * (d1 + dd + dd);
        let rbinf = cd * dd;
        sums[0] += la1 * rb1; // 1
        sums[1] += la2 * rb2; // 2
        sums[2] += la3 * rb3; // 3
        sums[3] += lainf * rbinf; // ∞
        return;
    }
    if D == 8 {
        // Reconstruct (v0,v1) on stack and use optimized kernel
        let mut arr: [(F, F); 8] = [(F::ZERO, F::ZERO); 8];
        for j in 0..8 {
            let v1 = pairs_v1_diff[j].0;
            let diff = pairs_v1_diff[j].1;
            arr[j] = (v1 - diff, v1);
        }
        eval_inter8_final_accumulate(&arr, sums);
        return;
    }
    if D == 16 {
        let mut arr: [(F, F); 16] = [(F::ZERO, F::ZERO); 16];
        for j in 0..16 {
            let v1 = pairs_v1_diff[j].0;
            let diff = pairs_v1_diff[j].1;
            arr[j] = (v1 - diff, v1);
        }
        eval_inter16_final_accumulate(&arr, sums);
        return;
    }
    if D == 32 {
        let mut arr: [(F, F); 32] = [(F::ZERO, F::ZERO); 32];
        for j in 0..32 {
            let v1 = pairs_v1_diff[j].0;
            let diff = pairs_v1_diff[j].1;
            arr[j] = (v1 - diff, v1);
        }
        eval_inter32_final_accumulate(&arr, sums);
        return;
    }
    // Fallback: reconstruct (v0,v1) and delegate
    let mut pairs: [(F, F); D] = [(F::ZERO, F::ZERO); D];
    for j in 0..D {
        let v1 = pairs_v1_diff[j].0;
        let diff = pairs_v1_diff[j].1;
        let v0 = v1 - diff;
        pairs[j] = (v0, v1);
    }
    product_eval_univariate_accumulate::<F, D>(&pairs, sums);
}



// -----------------------------------------------------------------------------
// General univariate extrapolation utilities (moved from multivariate.rs)
// -----------------------------------------------------------------------------

/// Univariate extrapolation from evaluations at nodes `x_nodes[0..=k]` to `y_nodes[0..=h]`.
/// Inputs represent a degree-<=k polynomial by its values on k+1 distinct nodes, and returns
/// its values on h+1 nodes. All nodes must be distinct within their own lists; `y_nodes` may
/// include values from `x_nodes` and will be evaluated consistently.
pub fn univariate_extrapolate<F: Field>(
    x_nodes: &[F],
    x_values: &[F],
    y_nodes: &[F],
) -> Vec<F> {
    assert_eq!(x_nodes.len(), x_values.len(), "nodes/values length mismatch");
    let k = x_nodes.len();
    assert!(k >= 2, "need at least 2 points");

    // Precompute barycentric weights w_i = 1 / prod_{j!=i} (x_i - x_j)
    let mut weights: Vec<F> = vec![F::ZERO; k];
    for i in 0..k {
        let mut denom = F::ONE;
        let xi = x_nodes[i];
        for j in 0..k {
            if i == j { continue; }
            denom *= xi - x_nodes[j];
        }
        weights[i] = denom.inverse().expect("distinct x_nodes required");
    }

    // Evaluate using first-form barycentric interpolation
    let mut out = Vec::with_capacity(y_nodes.len());
    for &y in y_nodes.iter() {
        // Check if y equals an existing node to avoid division by zero
        let mut at_node: Option<usize> = None;
        for i in 0..k {
            if y == x_nodes[i] { at_node = Some(i); break; }
        }
        if let Some(i) = at_node {
            out.push(x_values[i]);
            continue;
        }

        let mut num = F::ZERO;
        let mut den = F::ZERO;
        for i in 0..k {
            let li = weights[i] / (y - x_nodes[i]);
            num += li * x_values[i];
            den += li;
        }
        out.push(num * den.inverse().expect("nonzero denominator"));
    }
    out
}
