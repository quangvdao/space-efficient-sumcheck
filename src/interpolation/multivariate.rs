use ark_ff::Field;
use crate::interpolation::univariate::extrapolate_uk_to_uh_canonical;
use crate::interpolation::field_mul_small::FieldMulSmall;

/// Compute row-major strides for a given `shape` (last axis contiguous).
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let v = shape.len();
    let mut strides = vec![1usize; v];
    for i in (0..v.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Trait to allow pluggable univariate extrapolation implementations.
pub trait UniExtrap<F: Field> {
    fn extrapolate(x_nodes: &[F], x_values: &[F], y_nodes: &[F]) -> Vec<F>;
}

/// Multivariate extrapolation on arbitrary finite nodes (no ∞):
///
/// Given evaluations on a grid U_k^v specified by per-axis nodes `axes_k[j]` (length k+1 each),
/// extend to evaluations on U_d^v given `axes_d[j]` by repeatedly applying a univariate
/// extrapolator along each axis. The input `values` is a flat array in row-major order with
/// last axis contiguous.
///
/// Paper alignment (Sec. 4 / App. G): this is Procedure MultiExtrap specialized to finite nodes.
/// It performs v passes; in pass j it extrapolates along axis j for all slices S of the other
/// v-1 coordinates. The number of univariate calls equals
///   N_tot = ((d+1)^v - (k+1)^v) / (d - k).
/// For the product recursion’s doubling regime and v≥2, this work is lower-order and contributes
/// only small-by-big ops and additions in the paper’s cost model.
// Old generic finite-node multivariate extrapolator removed.

/// Generic version of multivariate extrapolation using a provided univariate extrapolation strategy.
// Old generic strategy-based multivariate extrapolator removed.

// (univariate_extrapolate_nodes implementation moved to interpolation::univariate)

/// Multivariate extrapolation using node sets that may include Infinity at the end of each axis.
///
/// Matches the paper’s MultiExtrap but uses node sets U_k = {1,2,...,k, ∞} commonly used in the
/// implementation to simplify fast paths. Each pass reduces to univariate extrapolation over the
/// selected axis, and we delegate per-line extrapolations to a node-aware strategy. The default
/// strategy uses `univariate_extrapolate_nodes`, which has a fast path for canonical {1..k,∞}→{1..h,∞}.
///
/// Cost: identical structure to the finite-nodes variant. Extrapolation introduces no big-by-big
/// multiplications; all big-by-big work arises later in pointwise products (see product eval).
pub fn multivariate_extrapolate_canonical<F: FieldMulSmall>(
    v: usize,
    k: usize,
    h: usize,
    values: &[F],
) -> Vec<F> {
    assert!(k >= 1 && k <= 16 && h >= k && h <= 32);
    let mut cur = values.to_vec();
    let mut shape: Vec<usize> = core::iter::repeat(k + 1).take(v).collect();
    for axis in 0..v {
        if k == h { break; }
        let mut strides_src = vec![1usize; v];
        for i in (0..v.saturating_sub(1)).rev() { strides_src[i] = strides_src[i + 1] * shape[i + 1]; }
        let mut next_shape = shape.clone();
        next_shape[axis] = h + 1;
        let mut strides_dst = vec![1usize; v];
        for i in (0..v.saturating_sub(1)).rev() { strides_dst[i] = strides_dst[i + 1] * next_shape[i + 1]; }
        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() { if i != axis { num_slices *= *s; } }
        let src_axis_len = shape[axis];
        let dst_axis_len = next_shape[axis];
        let total_next_elems = next_shape.iter().copied().fold(1usize, |acc, s| acc * s);
        let mut next = vec![F::ZERO; total_next_elems];
        for slice_idx in 0..num_slices {
            let mut rem = slice_idx;
            let mut src_base = 0usize;
            let mut dst_base = 0usize;
            for i in 0..v {
                if i == axis { continue; }
                let si = shape[i];
                let coord = rem % si;
                rem /= si;
                src_base += coord * strides_src[i];
                dst_base += coord * strides_dst[i];
            }
            let mut line: Vec<F> = Vec::with_capacity(src_axis_len);
            for t in 0..src_axis_len { line.push(cur[src_base + t * strides_src[axis]]); }
            let line_ex = extrapolate_uk_to_uh_canonical::<F>(&line, h);
            for t in 0..dst_axis_len { next[dst_base + t * strides_dst[axis]] = line_ex[t]; }
        }
        cur = next;
        shape = next_shape;
    }
    cur
}

#[inline]
fn lift_01_to_u1_grid<F: Field>(v: usize, values_01: &[F]) -> Vec<F> {
    // values_01: length 2^v, last axis contiguous
    let mut cur = values_01.to_vec();
    let shape: Vec<usize> = core::iter::repeat(2usize).take(v).collect();
    for axis in 0..v {
        let strides = compute_strides(&shape);
        let mut next = vec![F::ZERO; cur.len()];
        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() { if i != axis { num_slices *= *s; } }
        for slice_idx in 0..num_slices {
            let mut rem = slice_idx;
            let mut base = 0usize;
            for i in 0..v {
                if i == axis { continue; }
                let si = shape[i];
                let coord = rem % si;
                rem /= si;
                base += coord * strides[i];
            }
            let f0 = cur[base + 0 * strides[axis]];
            let f1 = cur[base + 1 * strides[axis]];
            next[base + 0 * strides[axis]] = f1;         // 1
            next[base + 1 * strides[axis]] = f1 - f0;     // ∞
        }
        cur = next;
    }
    cur
}

/// Canonical multivariate product: inputs are D polynomials on {0,1}^v; returns product on U_d^v
/// using optimized univariate canonical extrapolation.
pub fn multivariate_product_evaluations_canonical<F: FieldMulSmall>(
    v: usize,
    polys_01: &[Vec<F>],
    d: usize,
) -> Vec<F> {
    assert!(v >= 1);
    assert!(!polys_01.is_empty());
    assert_eq!(polys_01.len(), d);
    let expected_len = 1usize << v;
    for p in polys_01 { assert_eq!(p.len(), expected_len); }

    fn rec<F: FieldMulSmall>(v: usize, polys: &[Vec<F>], deg_parent: usize) -> Vec<F> {
        if polys.len() == 1 {
            let u1 = lift_01_to_u1_grid::<F>(v, &polys[0]);
            return multivariate_extrapolate_canonical::<F>(v, 1, deg_parent, &u1);
        }
        let d_here = polys.len();
        let m = d_here / 2;
        let left = rec::<F>(v, &polys[..m], m);
        let right = rec::<F>(v, &polys[m..], d_here - m);
        let left_ex = multivariate_extrapolate_canonical::<F>(v, m, deg_parent, &left);
        let right_ex = multivariate_extrapolate_canonical::<F>(v, d_here - m, deg_parent, &right);
        let mut out = left_ex;
        for (o, r) in out.iter_mut().zip(right_ex.iter()) { *o *= *r; }
        out
    }

    rec::<F>(v, polys_01, d)
}

/// Canonical multivariate product accumulate variant: adds product on U_d^v into `sums`.
/// Precondition: sums.len() == (d+1)^v.
pub fn multivariate_product_evaluations_canonical_accumulate<F: FieldMulSmall>(
    v: usize,
    polys_01: &[Vec<F>],
    d: usize,
    sums: &mut [F],
) {
    assert!(v >= 1);
    assert!(!polys_01.is_empty());
    assert_eq!(polys_01.len(), d);
    let expected_len = 1usize << v;
    for p in polys_01 { assert_eq!(p.len(), expected_len); }
    let out_len = (d + 1).pow(v as u32) as usize;
    assert_eq!(sums.len(), out_len);

    fn rec_acc<F: FieldMulSmall>(v: usize, polys: &[Vec<F>], deg_parent: usize, sums: &mut [F]) {
        if polys.len() == 1 {
            let u1 = lift_01_to_u1_grid::<F>(v, &polys[0]);
            let vals = multivariate_extrapolate_canonical::<F>(v, 1, deg_parent, &u1);
            for (s, p) in sums.iter_mut().zip(vals.iter()) { *s += *p; }
            return;
        }
        let d_here = polys.len();
        let m = d_here / 2;
        // Compute left product on U_m^v and lift to U_{deg_parent}^v once
        let left_prod_m = multivariate_product_evaluations_canonical::<F>(v, &polys[..m], m);
        let left_ex = multivariate_extrapolate_canonical::<F>(v, m, deg_parent, &left_prod_m);
        // Compute right product on U_{d_here-m}^v and lift to U_{deg_parent}^v
        let right_prod = multivariate_product_evaluations_canonical::<F>(v, &polys[m..], d_here - m);
        let right_ex = multivariate_extrapolate_canonical::<F>(v, d_here - m, deg_parent, &right_prod);
        // Accumulate pointwise product into sums
        for i in 0..sums.len() { sums[i] += left_ex[i] * right_ex[i]; }
    }

    rec_acc::<F>(v, polys_01, d, sums)
}
 


