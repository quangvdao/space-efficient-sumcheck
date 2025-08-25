use ark_ff::Field;
use ark_std::vec::Vec;

/// Equality polynomial helpers (non-split and split variants)
///
/// Provides:
/// - Full table evaluation eq(w, x) over x ∈ {0,1}^ℓ (and memoized prefixes)
/// - Split (k=2) variant producing two half tables for suffix handling
/// - Streaming DP recurrences for eq(r, x) traversal (g_i, h_i ratios)
///
/// These functions mirror the math in sections 6 and Appendix A: proc:eq-eval, proc:eq-eval-memoized,
/// and the streaming recurrence for eq(r, ·). Naming avoids external provenance.

/// Compute full table v where v[idx(x)] = eq(w, x) for x∈{0,1}^ℓ.
pub fn eq_table_full<F: Field>(w: &[F]) -> Vec<F> {
    let ell = w.len();
    let mut v: Vec<F> = vec![F::ONE];
    let mut s = 1usize;
    for i in 0..ell {
        let wi = w[i];
        let mut next = vec![F::ZERO; s * 2];
        for j in 0..s {
            let base = v[j];
            next[2 * j + 1] = base * wi;           // x_i = 1
            next[2 * j] = base - next[2 * j + 1];  // x_i = 0: (1-w_i) * base
        }
        v = next;
        s <<= 1;
    }
    v
}

/// Compute memoized prefix tables V_i such that V_i[idx(x)] = eq(w[0..i], x) for i=1..ℓ.
pub fn eq_prefix_tables<F: Field>(w: &[F]) -> Vec<Vec<F>> {
    let ell = w.len();
    let mut out: Vec<Vec<F>> = Vec::with_capacity(ell);
    let mut prev: Vec<F> = vec![F::ONE];
    let mut s = 1usize;
    for i in 0..ell {
        let wi = w[i];
        let mut cur = vec![F::ZERO; s * 2];
        for j in 0..s {
            let base = prev[j];
            cur[2 * j + 1] = base * wi;
            cur[2 * j] = base - cur[2 * j + 1];
        }
        out.push(cur.clone());
        prev = cur;
        s <<= 1;
    }
    out
}

/// Streaming ratios for eq(r, x) DP traversal: returns (g_i, h_i) arrays for i in 0..ℓ.
/// g_i = r_i * (1-r_i)^{-1}, h_i = (1-r_i) * r_i^{-1}.
pub fn eq_streaming_ratios<F: Field>(r: &[F]) -> (Vec<F>, Vec<F>) {
    let ell = r.len();
    let mut g = vec![F::ZERO; ell];
    let mut h = vec![F::ZERO; ell];
    for i in 0..ell {
        let one_minus = F::ONE - r[i];
        let inv_one_minus = one_minus.inverse().expect("nonzero 1-r");
        let inv_r = r[i].inverse().expect("nonzero r");
        g[i] = r[i] * inv_one_minus;
        h[i] = one_minus * inv_r;
    }
    (g, h)
}

/// Split (k=2) suffix tables: for a suffix w_s of length s, return two tables of sizes 2^{⌊s/2⌋} and 2^{⌈s/2⌉}.
/// The split starts at w_s[0]. Caller chooses how to map indices to bits.
pub fn eq_split_tables<F: Field>(w_suffix: &[F]) -> (Vec<F>, Vec<F>) {
    let s = w_suffix.len();
    if s == 0 { return (vec![], vec![]); }
    let s_left = s / 2;
    let s_right = s - s_left;
    let mut left = vec![F::ZERO; 1usize << s_left];
    let mut right = vec![F::ZERO; 1usize << s_right];
    left[0] = F::ONE; right[0] = F::ONE;
    for i in 0..s_left {
        let wi = w_suffix[i];
        let size = 1usize << i;
        for mask in 0..size {
            let base = left[mask];
            left[mask] = base * (F::ONE - wi);
            left[mask | size] = base * wi;
        }
    }
    for i in 0..s_right {
        let wi = w_suffix[s_left + i];
        let size = 1usize << i;
        for mask in 0..size {
            let base = right[mask];
            right[mask] = base * (F::ONE - wi);
            right[mask | size] = base * wi;
        }
    }
    (left, right)
}

/// Split evaluation (two halves) at a given pair of indices (idx_left, idx_right).
#[inline]
pub fn eq_split_eval_at<F: Field>(left: &[F], right: &[F], idx_left: usize, idx_right: usize) -> F {
    left[idx_left] * right[idx_right]
}

/// Evaluate eq(w_i, z) on U_d nodes for the linear factor in current round.
/// Returns values for z ∈ {1,2,...,D-1} and the leading coefficient at ∞.
/// Mapping: ell(1)=w, ell(z)=(1-w)+(2w-1)z for z≥1; ell(∞)=2w-1.
pub fn eq_linear_on_u_d<F: Field>(w_i: F, d: usize) -> (Vec<F>, F) {
    if d == 1 {
        return (vec![w_i], w_i + w_i - F::ONE);
    }
    let mut vals = vec![F::ZERO; d - 1];
    let slope = w_i + w_i - F::ONE; // 2w-1
    // z=1
    vals[0] = w_i;
    // z=2..d-1
    let mut lz = (F::ONE - w_i) + slope + slope; // (1-w) + (2w-1)*2
    for z in 2..d {
        vals[z - 1] = lz;
        lz += slope;
    }
    (vals, slope)
}
