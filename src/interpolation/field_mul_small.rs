use ark_ff::{Field, Zero, BigInteger};

/// Small-multiples helper trait, modeled after high-d-opt's FieldMulSmall.
///
/// Default implementations work for any Field via double-and-add; concrete
/// fields can provide faster overrides by implementing this trait.
pub trait FieldMulSmall: Field + Sized + Copy {
    #[inline]
    fn double(x: &Self) -> Self { *x + *x }

    // Small-multiple ops
    fn mul_u64(self, n: u64) -> Self;
    fn mul_i64(self, n: i64) -> Self;
    fn mul_u128(self, n: u128) -> Self;
    fn mul_i128(self, n: i128) -> Self;

    // Fast constructors / linear combos
    fn from_u64(n: u64) -> Self;
    fn linear_combination_u64(pairs: &[(Self, u64)]) -> Self;
    fn linear_combination_i64(pos: &[(Self, u64)], neg: &[(Self, u64)]) -> Self;
}

// BN254 specialization: delegate to ark-ff intrinsics (same as high-d-opt)
impl FieldMulSmall for ark_bn254::Fr {
    #[inline(always)]
    fn mul_u64(self, n: u64) -> Self {
        ark_ff::Fp::mul_u64::<5>(self, n)
    }

    #[inline(always)]
    fn mul_i64(self, n: i64) -> Self {
        ark_ff::Fp::mul_i64::<5>(self, n)
    }

    #[inline(always)]
    fn mul_u128(self, n: u128) -> Self {
        ark_ff::Fp::mul_u128::<5, 6>(self, n)
    }

    #[inline(always)]
    fn mul_i128(self, n: i128) -> Self {
        ark_ff::Fp::mul_i128::<5, 6>(self, n)
    }

    #[inline]
    fn from_u64(n: u64) -> Self {
        <Self as ark_ff::PrimeField>::from_u64::<5>(n).unwrap()
    }

    #[inline]
    fn linear_combination_u64(pairs: &[(Self, u64)]) -> Self {
        // Unreduced accumulation in BigInt, then one reduction
        let mut tmp = ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&pairs[0].0 .0, pairs[0].1);
        for (a, b) in &pairs[1..] {
            let carry = tmp.add_with_carry(&ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&a.0, *b));
            debug_assert!(!carry, "carry in linear_combination_u64");
        }
        ark_ff::Fp::from_unchecked_nplus1(tmp)
    }

    #[inline]
    fn linear_combination_i64(pos: &[(Self, u64)], neg: &[(Self, u64)]) -> Self {
        // Unreduced pos sum
        let mut pos_lc = ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&pos[0].0 .0, pos[0].1);
        for (a, b) in &pos[1..] {
            let carry = pos_lc.add_with_carry(&ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&a.0, *b));
            debug_assert!(!carry, "carry in linear_combination_i64(+)");
        }
        // Unreduced neg sum
        let mut neg_lc = ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&neg[0].0 .0, neg[0].1);
        for (a, b) in &neg[1..] {
            let carry = neg_lc.add_with_carry(&ark_ff::BigInt::<4>::mul_u64_w_carry::<5>(&a.0, *b));
            debug_assert!(!carry, "carry in linear_combination_i64(-)");
        }
        // Subtract and reduce once
        match pos_lc.cmp(&neg_lc) {
            core::cmp::Ordering::Greater => {
                let borrow = pos_lc.sub_with_borrow(&neg_lc);
                debug_assert!(!borrow, "borrow in linear_combination_i64");
                ark_ff::Fp::from_unchecked_nplus1(pos_lc)
            }
            core::cmp::Ordering::Less => {
                let borrow = neg_lc.sub_with_borrow(&pos_lc);
                debug_assert!(!borrow, "borrow in linear_combination_i64");
                -ark_ff::Fp::from_unchecked_nplus1(neg_lc)
            }
            core::cmp::Ordering::Equal => ark_ff::Fp::zero(),
        }
    }
}


