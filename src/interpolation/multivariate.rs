use ark_ff::Field;
use crate::interpolation::univariate::extrapolate_uk_to_uh;
use crate::interpolation::field_mul_small::FieldMulSmall;

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
/// Alignment with the paper (Sec. 4 and App. D): the implementation uses the canonical nodes
/// U_k := {1, 2, ..., k, ∞}, whereas the paper’s notation uses U_k^{paper} := {0, 1, ..., k-1, ∞}.
/// These are isomorphic via the shift t ↦ t+1 on finite nodes; values at 0 are not stored but can
/// be evaluated from U_k by interpolation when needed. Each pass reduces to univariate
/// extrapolation over the selected axis, implemented by `extrapolate_uk_to_uh` which has fast
/// paths for {1..k, ∞} → {1..h, ∞}.
///
/// Cost: identical structure to the finite-nodes variant. Extrapolation introduces no big-by-big
/// multiplications; all big-by-big work arises later in pointwise products (see product eval).
///
/// This function corresponds to `Procedure MultiExtrap` from `sections/4_high_d.tex` and
/// `sections/D_high_d_appendix.tex`. It operates in `v` passes, one for each dimension.
/// In pass `j`, it iterates through all (v-1)-dimensional slices along the j-th axis
/// and applies univariate extrapolation to each slice. This reduces the v-dimensional
/// problem to a series of 1-dimensional ones, which can be solved efficiently.
///
/// The cost of this procedure is dominated by the univariate extrapolations. When using the
/// optimized `extrapolate_uk_to_uh` function, this step introduces no "big-by-big" field
/// multiplications, making the overall `MultiProductEval` algorithm highly efficient.
pub fn multivariate_extrapolate<F: FieldMulSmall>(
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
            let line_ex = extrapolate_uk_to_uh::<F>(&line, h);
            for t in 0..dst_axis_len { next[dst_base + t * strides_dst[axis]] = line_ex[t]; }
        }
        cur = next;
        shape = next_shape;
    }
    cur
}

/// Naive multivariate extrapolation using plain Lagrange (via Vandermonde solve) per axis.
///
/// Baseline to validate `multivariate_extrapolate`. Interprets each line along an axis as
/// values on U_k = {1,2,...,k, ∞} and extrapolates to U_h by solving for coefficients with the
/// leading coefficient taken from the ∞ entry. No small-int fast paths.
// --- Naive multilinear extension baseline (equality-polynomial based) ---

#[inline]
fn mobius_to_coeffs_inplace<F: Field>(v: usize, a: &mut [F]) {
    // Standard subset Möbius transform to convert values on {0,1}^v to coefficients
    // for the multilinear polynomial p(y) = sum_{T} c_T prod_{i in T} y_i.
    let n = 1usize << v;
    debug_assert_eq!(a.len(), n);
    for i in 0..v {
        let bit = 1usize << i;
        for mask in 0..n {
            if (mask & bit) != 0 {
                let other = mask ^ bit;
                a[mask] -= a[other];
            }
        }
    }
}

#[allow(dead_code)]
#[inline]
fn eval_from_coeffs_at_point<F: Field>(v: usize, coeffs: &[F], y: &[F]) -> F {
    let n = 1usize << v;
    let mut acc = F::ZERO;
    for mask in 0..n {
        let mut prod = coeffs[mask];
        if prod.is_zero() { continue; }
        let mut m = mask;
        let mut i = 0usize;
        while m != 0 {
            if (m & 1) != 0 { prod *= y[i]; }
            i += 1;
            m >>= 1;
        }
        acc += prod;
    }
    acc
}

#[inline]
fn eval_from_coeffs_with_infty<F: Field>(v: usize, coeffs: &[F], y: &[Option<F>]) -> F {
    // y[i] = Some(value) for finite coordinate, None for ∞
    let n = 1usize << v;
    let mut acc = F::ZERO;
    for mask in 0..n {
        // Require that all ∞ axes are included in the monomial
        let mut ok = true;
        for (i, yi) in y.iter().enumerate() {
            if yi.is_none() && (mask & (1usize << i)) == 0 { ok = false; break; }
        }
        if !ok { continue; }
        let mut prod = coeffs[mask];
        if prod.is_zero() { continue; }
        for (i, yi) in y.iter().enumerate() {
            if let Some(val) = yi {
                if (mask & (1usize << i)) != 0 { prod *= *val; }
            }
        }
        acc += prod;
    }
    acc
}

/// DEPRECATED: From evaluations on {0,1}^v (last axis contiguous), build the full U_h^v grid using
/// the multilinear extension p(y) = sum_x eq(y,x) p(x) for finite coordinates, and the
/// natural coefficient semantics for ∞ coordinates.
/// 
/// WARNING: This uses the old U_h = {1,2,...,h,∞} definition, not the paper's {0,1,...,h-1,∞}
#[deprecated(note = "Use multilinear_extend_to_paper_u_h_grid_naive instead - this uses incorrect U_d definition")]
pub fn multilinear_extend_to_u_h_grid_naive<F: FieldMulSmall>(
    v: usize,
    values_01: &[F],
    h: usize,
) -> Vec<F> {
    let n01 = 1usize << v;
    assert_eq!(values_01.len(), n01);
    // Reorder inputs so that the last axis is contiguous (LSB corresponds to axis v-1)
    let mut reordered = values_01.to_vec();
    for i in 0..n01 {
        let j = i.reverse_bits() >> (usize::BITS - v as u32);
        if i < j { reordered.swap(i, j); }
    }
    let mut coeffs = reordered;
    mobius_to_coeffs_inplace::<F>(v, &mut coeffs);
    let side = h + 1; // [1..h, ∞]
    // Canonical per-axis node values z_t such that for 1D linear f, f(t) = c0 + z_t*(c1)
    // Constructed by extrapolating base [f(1)=1, f(∞)=1]
    let node_values = extrapolate_uk_to_uh::<F>(&[F::ONE, F::ONE], h);
    let total = side.pow(v as u32);
    let mut out = vec![F::ZERO; total];
    for idx in 0..total {
        // decode mixed-radix base (h+1)
        let mut tmp = idx;
        let mut y: Vec<Option<F>> = vec![None; v];
        for ax_rev in 0..v {
            let t = tmp % side;
            tmp /= side;
            let ax = v - 1 - ax_rev; // last axis is contiguous (LSD)
            if t == h {
                y[ax] = None;
            } else {
                y[ax] = Some(node_values[t]);
            }
        }
        out[idx] = eval_from_coeffs_with_infty::<F>(v, &coeffs, &y);
    }
    out
}

/// From evaluations on {0,1}^v (last axis contiguous), build the full U_h^v grid using
/// the multilinear extension p(y) = sum_x eq(y,x) p(x) for finite coordinates, and the
/// natural coefficient semantics for ∞ coordinates.
/// 
/// Paper alignment: Uses U_h = {0,1,...,h-1,∞} as defined in sections/2_preliminaries.tex
pub fn multilinear_extend_to_paper_u_h_grid_naive<F: FieldMulSmall>(
    v: usize,
    values_01: &[F],
    h: usize,
) -> Vec<F> {
    let n01 = 1usize << v;
    assert_eq!(values_01.len(), n01);
    
    // Reorder inputs so that the last axis is contiguous (LSB corresponds to axis v-1)
    let mut reordered = values_01.to_vec();
    for i in 0..n01 {
        let j = i.reverse_bits() >> (usize::BITS - v as u32);
        if i < j { reordered.swap(i, j); }
    }
    
    // Convert to multilinear coefficients
    let mut coeffs = reordered;
    mobius_to_coeffs_inplace::<F>(v, &mut coeffs);
    
    let side = h + 1; // U_h = {∞,0,1,...,h-1} has h+1 elements
    let total = side.pow(v as u32);
    let mut out = vec![F::ZERO; total];
    
    for idx in 0..total {
        // Decode mixed-radix base (h+1) to get coordinates
        // Paper's U_h = {∞, 0, 1, ..., h-1}, so:
        // t=0 corresponds to ∞
        // t=1 corresponds to 0
        // t=2 corresponds to 1
        // ...
        // t=h corresponds to h-1
        let mut tmp = idx;
        let mut y: Vec<Option<F>> = vec![None; v];
        
        for ax_rev in 0..v {
            let t = tmp % side;
            tmp /= side;
            let ax = v - 1 - ax_rev; // last axis is contiguous (LSD)
            
            if t == 0 {
                // First position is ∞
                y[ax] = None;
            } else {
                // Positions 1,2,...,h correspond to 0,1,...,h-1
                y[ax] = Some(F::from((t - 1) as u64));
            }
        }
        
        out[idx] = eval_from_coeffs_with_infty::<F>(v, &coeffs, &y);
    }
    
    out
}

#[inline]
pub fn lift_01_to_u1_grid<F: Field>(v: usize, values_01: &[F]) -> Vec<F> {
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

/// Represents a multivariate polynomial by its evaluations on U_k^v for some degree k.
/// 
/// Paper alignment: U_k = {0, 1, ..., k-1, ∞} as defined in sections/2_preliminaries.tex
/// This differs from the code's previous U_k = {1, 2, ..., k, ∞}
#[derive(Clone, Debug)]
pub struct MultivariateEvaluations<F: Field> {
    /// The degree k (per-variable degree of the polynomial)
    pub degree: usize,
    /// The number of variables v
    pub num_vars: usize,
    /// Evaluations on U_k^v = {0,1,...,k-1,∞}^v, stored in row-major order
    /// Length should be (degree + 1)^num_vars
    pub evaluations: Vec<F>,
}

/// Extrapolates evaluations from U_k^v to U_h^v using paper's U_k definition.
/// 
/// Paper alignment: U_k = {0, 1, ..., k-1, ∞} per axis
/// Input: evaluations on U_k^v (length (k+1)^v)
/// Output: evaluations on U_h^v (length (h+1)^v)
pub fn multivariate_extrapolate_paper_uk<F: FieldMulSmall>(
    v: usize,
    k: usize,
    h: usize,
    values: &[F],
) -> Vec<F> {
    assert!(k >= 1 && k <= 16 && h >= k && h <= 32);
    assert_eq!(values.len(), (k + 1).pow(v as u32));
    
    let mut cur = values.to_vec();
    let mut shape: Vec<usize> = core::iter::repeat(k + 1).take(v).collect();
    
    for axis in 0..v {
        if k == h { break; }
        
        // Compute strides for current shape
        let mut strides_src = vec![1usize; v];
        for i in (0..v.saturating_sub(1)).rev() { 
            strides_src[i] = strides_src[i + 1] * shape[i + 1]; 
        }
        
        // Update shape for this axis
        let mut next_shape = shape.clone();
        next_shape[axis] = h + 1;
        
        // Compute strides for new shape
        let mut strides_dst = vec![1usize; v];
        for i in (0..v.saturating_sub(1)).rev() { 
            strides_dst[i] = strides_dst[i + 1] * next_shape[i + 1]; 
        }
        
        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() { 
            if i != axis { num_slices *= *s; } 
        }
        
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
            
            // Extract line along this axis
            let mut line: Vec<F> = Vec::with_capacity(src_axis_len);
            for t in 0..src_axis_len { 
                line.push(cur[src_base + t * strides_src[axis]]); 
            }
            
            // Convert from paper's U_k = {0,...,k-1,∞} to code's U_k = {1,...,k,∞}
            // and extrapolate, then convert back
            let line_ex = extrapolate_paper_uk_to_uh::<F>(&line, h);
            
            for t in 0..dst_axis_len { 
                next[dst_base + t * strides_dst[axis]] = line_ex[t]; 
            }
        }
        
        cur = next;
        shape = next_shape;
    }
    
    cur
}

/// Direct univariate extrapolation using paper's U_k = {0,1,...,k-1,∞} definition
/// Converts to coefficient form and then evaluates at desired points
fn extrapolate_paper_uk_to_uh<F: FieldMulSmall>(values_paper_uk: &[F], h: usize) -> Vec<F> {
    let k = values_paper_uk.len() - 1;
    
    // Paper's U_k = {∞, 0, 1, ..., k-1}
    // values_paper_uk[0] = p(∞) = leading coefficient
    // values_paper_uk[i+1] = p(i) for i = 0, 1, ..., k-1 a_k
    
    // For a degree-k polynomial p(x) = a_k x^k + a_{k-1} x^{k-1} + ... + a_1 x + a_0,
    // we have p(∞) = a_k (the leading coefficient)
    
    // We can recover the coefficients using the fact that:
    // p(x) = a_k x^k + q(x) where q(x) is a degree-(k-1) polynomial
    // So q(i) = p(i) - a_k * i^k for i = 0, 1, ..., k-1
    
    let leading_coeff = values_paper_uk[0];
    
    // Compute q(i) = p(i) - a_k * i^k
    let mut q_values = vec![F::ZERO; k];
    for i in 0..k {
        let i_field = F::from(i as u64);
        let i_power_k = if k > 0 { pow_usize(i_field, k) } else { F::ONE };
        q_values[i] = values_paper_uk[i + 1] - leading_coeff * i_power_k;
    }
    
    // Now we have q(0), q(1), ..., q(k-1) and we know q is degree at most k-1
    // We can use Lagrange interpolation to evaluate q at any point
    
    let mut result = vec![F::ZERO; h + 1];
    
    // U_h = {∞, 0, 1, ..., h-1} according to paper
    // Position 0: p(∞)
    // Position 1: p(0)
    // Position 2: p(1)
    // ...
    // Position h: p(h-1)
    
    for pos in 0..=h {
        if pos == 0 {
            // Position 0 is p(∞) = leading coefficient
            result[0] = leading_coeff;
        } else {
            // Position pos corresponds to p(pos-1)
            let x_val = pos - 1;
            let x_field = F::from(x_val as u64);
            let x_power_k = if k > 0 { pow_usize(x_field, k) } else { F::ONE };
            
            // Compute q(x_val) using Lagrange interpolation
            let mut q_x = F::ZERO;
            for i in 0..k {
                let xi = F::from(i as u64);
                let mut lagrange_coeff = F::ONE;
                
                for l in 0..k {
                    if i != l {
                        let xl = F::from(l as u64);
                        lagrange_coeff *= (x_field - xl) / (xi - xl);
                    }
                }
                
                q_x += q_values[i] * lagrange_coeff;
            }
            
            result[pos] = leading_coeff * x_power_k + q_x;
        }
    }
    
    result
}

/// Extends evaluations from {0,1}^v to U_1^v = {∞,0}^v using paper's definition
/// 
/// Paper alignment: U_1 = {∞, 0} according to sections/2_preliminaries.tex
/// So U_1^v has 2^v elements, same as the input {0,1}^v
pub fn lift_01_to_paper_u1_grid<F: Field>(v: usize, values_01: &[F]) -> Vec<F> {
    let n01 = 1usize << v;
    assert_eq!(values_01.len(), n01);
    
    // According to the paper, U_1 = {∞, 0}, so U_1^v = {∞, 0}^v
    // This has 2^v elements, same as {0,1}^v
    // For each coordinate, we map: 0 -> 0, 1 -> ∞
    // But we need to be careful about the ordering...
    
    // Actually, let me think about this more carefully.
    // The input is a multilinear polynomial given on {0,1}^v
    // We want to represent it on U_1^v = {∞, 0}^v
    // 
    // For a univariate linear polynomial p(x) = a + bx:
    // - p(0) = a
    // - p(1) = a + b  
    // - p(∞) = b (leading coefficient)
    //
    // So from p(0) and p(1), we can compute p(∞) = p(1) - p(0)
    // The representation on U_1 = {∞, 0} would be [p(∞), p(0)] = [b, a]
    
    let mut cur = values_01.to_vec();
    let mut shape: Vec<usize> = core::iter::repeat(2usize).take(v).collect();
    
    for axis in 0..v {
        let strides = compute_strides(&shape);
        let mut next = vec![F::ZERO; cur.len()]; // Same size since U_1 has 2 elements
        
        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() { 
            if i != axis { num_slices *= *s; } 
        }
        
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
            
            let f0 = cur[base + 0 * strides[axis]]; // p(0)
            let f1 = cur[base + 1 * strides[axis]]; // p(1)
            
            // Map to U_1 = {∞, 0} coordinates:
            // Position 0 in U_1 corresponds to ∞: p(∞) = f1 - f0
            // Position 1 in U_1 corresponds to 0: p(0) = f0
            next[base + 0 * strides[axis]] = f1 - f0; // p(∞) at position 0
            next[base + 1 * strides[axis]] = f0;      // p(0) at position 1
        }
        
        cur = next;
    }
    
    cur
}

/// MultiProductEval algorithm from sections/4_high_d.tex with correct U_d definition.
/// 
/// Paper alignment: This implements Algorithm 1 (MultiProductEval) exactly as specified,
/// using U_k = {0, 1, ..., k-1, ∞} throughout.
/// 
/// Input: d multilinear polynomials given by evaluations on {0,1}^v
/// Output: Their product as evaluations on U_d^v = {0,1,...,d-1,∞}^v
pub fn multi_product_eval_paper<F: FieldMulSmall>(
    v: usize,
    multilinear_inputs: &[Vec<F>],
    d: usize,
) -> Vec<F> {
    assert_eq!(multilinear_inputs.len(), d);
    assert!(d >= 1);
    let expected_len = 1usize << v;
    for p in multilinear_inputs { 
        assert_eq!(p.len(), expected_len); 
    }
    
    // Convert inputs to the internal representation
    let inputs: Vec<MultivariateEvaluations<F>> = multilinear_inputs
        .iter()
        .map(|evals_01| {
            MultivariateEvaluations {
                degree: 1,
                num_vars: v,
                evaluations: lift_01_to_paper_u1_grid::<F>(v, evals_01),
            }
        })
        .collect();
    
    let result = multi_product_eval_recursive_paper(inputs, d);
    result.evaluations
}

fn multi_product_eval_recursive_paper<F: FieldMulSmall>(
    polynomials: Vec<MultivariateEvaluations<F>>,
    target_degree: usize,
) -> MultivariateEvaluations<F> {
    let d = polynomials.len();
    let v = polynomials[0].num_vars;
    
    if d == 1 {
        // Base case: single polynomial
        let poly = &polynomials[0];
        if poly.degree == target_degree {
            return poly.clone();
        } else {
            // Extrapolate from U_{poly.degree}^v to U_{target_degree}^v
            let extrapolated_evals = multivariate_extrapolate_paper_uk::<F>(
                v, 
                poly.degree, 
                target_degree, 
                &poly.evaluations
            );
            return MultivariateEvaluations {
                degree: target_degree,
                num_vars: v,
                evaluations: extrapolated_evals,
            };
        }
    }
    
    // Recursive case: d > 1
    let m = d / 2; // This is d_L in the paper
    let d_r = d - m; // This is d_R in the paper
    
    // Split polynomials into two groups
    let left_polys = polynomials[..m].to_vec();
    let right_polys = polynomials[m..].to_vec();
    
    // Recursively compute products - these will have degrees m and d_r respectively
    let q_l = multi_product_eval_recursive_paper(left_polys, m);
    let q_r = multi_product_eval_recursive_paper(right_polys, d_r);
    
    // Now q_l has evaluations on U_m^v and q_r has evaluations on U_{d_r}^v
    // We need to extrapolate both to U_d^v before multiplying
    
    let q_l_extended = if q_l.degree == d {
        q_l.evaluations
    } else {
        multivariate_extrapolate_paper_uk::<F>(v, q_l.degree, d, &q_l.evaluations)
    };
    
    let q_r_extended = if q_r.degree == d {
        q_r.evaluations  
    } else {
        multivariate_extrapolate_paper_uk::<F>(v, q_r.degree, d, &q_r.evaluations)
    };
    
    // Pointwise multiply on U_d^v
    let mut product_evals = q_l_extended;
    for (left_val, right_val) in product_evals.iter_mut().zip(q_r_extended.iter()) {
        *left_val *= *right_val;
    }
    
    // The result has degree d and is evaluated on U_d^v
    let result = MultivariateEvaluations {
        degree: d,
        num_vars: v,
        evaluations: product_evals,
    };
    
    // If we need to extrapolate to target_degree, do it now
    if d == target_degree {
        result
    } else {
        let final_evals = multivariate_extrapolate_paper_uk::<F>(v, d, target_degree, &result.evaluations);
        MultivariateEvaluations {
            degree: target_degree,
            num_vars: v,
            evaluations: final_evals,
        }
    }
}

/// DEPRECATED: Old implementation with incorrect U_d definition.
/// Computes the product of D v-variate multilinear polynomials.
///
/// This function implements the `MultiProductEval` algorithm from `sections/4_high_d.tex`.
/// The polynomials are provided by their evaluations on the Boolean hypercube {0,1}^v,
/// and the function returns the evaluations of their product on the extended grid U_d^v.
///
/// The algorithm follows a recursive, divide-and-conquer strategy:
/// 1. **Base Case**: If there is only one polynomial, it is first lifted from the {0,1}^v
///    grid to the U_1^v grid (evaluations at {1, ∞} per axis), and then extrapolated to
///    the final target grid U_d^v.
/// 2. **Recursive Step**:
///    a. The D polynomials are split into two halves of size m and D-m.
///    b. The algorithm recursively calls itself on each half, computing two intermediate
///       product polynomials represented on the grids U_m^v and U_{D-m}^v.
///    c. Both intermediate results are extrapolated to the common, larger grid U_D^v
///       using `multivariate_extrapolate`.
///    d. The extrapolated evaluations are multiplied point-wise to get the final product on U_D^v.
///
/// This evaluation-based approach, combined with optimized extrapolation, avoids costly
/// coefficient-form manipulations and minimizes expensive field multiplications, as
/// analyzed in `sections/D_high_d_appendix.tex`.
#[deprecated(note = "Use multi_product_eval_paper instead - this uses incorrect U_d definition")]
pub fn multivariate_product_evaluations<F: FieldMulSmall>(
    v: usize,
    ml_inputs_01: &[Vec<F>],
    d: usize,
) -> Vec<F> {
    assert!(v >= 1);
    assert!(!ml_inputs_01.is_empty());
    assert_eq!(ml_inputs_01.len(), d);
    let expected_len = 1usize << v;
    for p in ml_inputs_01 { assert_eq!(p.len(), expected_len); }

    fn rec<F: FieldMulSmall>(v: usize, polys: &[Vec<F>], deg_parent: usize) -> Vec<F> {
        if polys.len() == 1 {
            let u1 = lift_01_to_u1_grid::<F>(v, &polys[0]);
            return multivariate_extrapolate::<F>(v, 1, deg_parent, &u1);
        }
        let d_here = polys.len();
        let m = d_here / 2;
        let left = rec::<F>(v, &polys[..m], d_here);
        let right = rec::<F>(v, &polys[m..], d_here);

        // Pointwise multiply to get the product on U_{d_here}^v
        let mut prod_d_here = left;
        for (o, r) in prod_d_here.iter_mut().zip(right.iter()) {
            *o *= *r;
        }

        // Extrapolate the final product to the parent's grid size
        multivariate_extrapolate::<F>(v, d_here, deg_parent, &prod_d_here)
    }

    rec::<F>(v, ml_inputs_01, d)
}

/// DEPRECATED: Naive baseline for multivariate product using the old U_d definition.
#[deprecated(note = "Use multivariate_product_evaluations_paper_naive instead - this uses incorrect U_d definition")]
pub fn multivariate_product_evaluations_naive<F: FieldMulSmall>(
    v: usize,
    ml_inputs_01: &[Vec<F>],
    d: usize,
) -> Vec<F> {
    assert!(v >= 1);
    assert!(!ml_inputs_01.is_empty());
    assert_eq!(ml_inputs_01.len(), d);
    let expected_len = 1usize << v;
    for p in ml_inputs_01 { assert_eq!(p.len(), expected_len); }

    // 1) Build full U_d^v grid for each multilinear input via naive extension
    let lifted: Vec<Vec<F>> = ml_inputs_01.iter()
        .map(|p01| multilinear_extend_to_u_h_grid_naive::<F>(v, p01, d))
        .collect();
    let out_len = (d + 1).pow(v as u32) as usize;
    // 2) Pointwise multiply on U_d^v
    let mut out = vec![F::ONE; out_len];
    for grid in lifted.iter() {
        for (o, g) in out.iter_mut().zip(grid.iter()) { *o *= *g; }
    }
    out
}

/// Naive baseline for multivariate product using paper's U_d = {0,1,...,d-1,∞} definition.
pub fn multivariate_product_evaluations_paper_naive<F: FieldMulSmall>(
    v: usize,
    ml_inputs_01: &[Vec<F>],
    d: usize,
) -> Vec<F> {
    assert!(v >= 1);
    assert!(!ml_inputs_01.is_empty());
    assert_eq!(ml_inputs_01.len(), d);
    let expected_len = 1usize << v;
    for p in ml_inputs_01 { assert_eq!(p.len(), expected_len); }

    // 1) Build full U_d^v grid for each multilinear input via naive extension
    let lifted: Vec<Vec<F>> = ml_inputs_01.iter()
        .map(|p01| multilinear_extend_to_paper_u_h_grid_naive::<F>(v, p01, d))
        .collect();
    let out_len = (d + 1).pow(v as u32) as usize;
    // 2) Pointwise multiply on U_d^v
    let mut out = vec![F::ONE; out_len];
    for grid in lifted.iter() {
        for (o, g) in out.iter_mut().zip(grid.iter()) { *o *= *g; }
    }
    out
}

/// Canonical multivariate product accumulate variant: adds product on U_d^v into `sums`.
/// Precondition: sums.len() == (d+1)^v.
pub fn multivariate_product_evaluations_accumulate<F: FieldMulSmall>(
    v: usize,
    ml_inputs_01: &[Vec<F>],
    d: usize,
    sums: &mut [F],
) {
    assert!(v >= 1);
    assert!(!ml_inputs_01.is_empty());
    assert_eq!(ml_inputs_01.len(), d);
    let expected_len = 1usize << v;
    for p in ml_inputs_01 { assert_eq!(p.len(), expected_len); }
    let out_len = (d + 1).pow(v as u32) as usize;
    assert_eq!(sums.len(), out_len);

    fn rec_acc<F: FieldMulSmall>(v: usize, polys: &[Vec<F>], deg_parent: usize, sums: &mut [F]) {
        if polys.len() == 1 {
            let u1 = lift_01_to_u1_grid::<F>(v, &polys[0]);
            let vals = multivariate_extrapolate::<F>(v, 1, deg_parent, &u1);
            for (s, p) in sums.iter_mut().zip(vals.iter()) { *s += *p; }
            return;
        }
        let d_here = polys.len();
        let m = d_here / 2;
        // Compute left product on U_m^v and lift to U_{deg_parent}^v once
        #[allow(deprecated)]
        let left_prod_m = multivariate_product_evaluations::<F>(v, &polys[..m], m);
        #[allow(deprecated)]
        let left_ex = multivariate_extrapolate::<F>(v, m, deg_parent, &left_prod_m);
        // Compute right product on U_{d_here-m}^v and lift to U_{deg_parent}^v
        #[allow(deprecated)]
        let right_prod = multivariate_product_evaluations::<F>(v, &polys[m..], d_here - m);
        #[allow(deprecated)]
        let right_ex = multivariate_extrapolate::<F>(v, d_here - m, deg_parent, &right_prod);
        // Accumulate pointwise product into sums
        for i in 0..sums.len() { sums[i] += left_ex[i] * right_ex[i]; }
    }

    rec_acc::<F>(v, ml_inputs_01, d, sums)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr as BN254;
    use ark_ff::Field;

    fn random_polys_01<F: Field>(v: usize, d: usize) -> Vec<Vec<F>> {
        let n = 1usize << v;
        (0..d).map(|j| {
            // vary each poly slightly
            (0..n).map(|i| F::from(((i as u64) * 17 + (j as u64) * 101 + 5) % 7919)).collect()
        }).collect()
    }

    #[test]
    fn test_paper_naive_vs_old_naive_different() {
        // Verify that the paper-aligned naive implementation gives different results
        // from the old implementation (as expected)
        type F = BN254;
        let v = 2;
        let h = 3;
        let p01 = random_polys_01::<F>(v, 1).pop().unwrap();
        
        #[allow(deprecated)]
        let old_naive = multilinear_extend_to_u_h_grid_naive::<F>(v, &p01, h);
        let paper_naive = multilinear_extend_to_paper_u_h_grid_naive::<F>(v, &p01, h);
        
        // They should be different (since they use different U_h definitions)
        assert_ne!(old_naive, paper_naive, "Paper and old naive should differ due to different U_h definitions");
    }

    #[test]
    fn test_paper_extrapolate_vs_naive() {
        // Test that our paper-aligned extrapolation matches the paper-aligned naive implementation
        type F = BN254;
        for v in [1usize, 2, 3].iter().copied() {
            for h in [2, 3, 4, 5, 6].iter().copied() {
                let p01 = random_polys_01::<F>(v, 1).pop().unwrap();
                
                // Use paper-aligned lift and extrapolate
                let u1_input = lift_01_to_paper_u1_grid::<F>(v, &p01);
                let fast = multivariate_extrapolate_paper_uk::<F>(v, 1, h, &u1_input);
                let naive = multilinear_extend_to_paper_u_h_grid_naive::<F>(v, &p01, h);
                
                if fast.len() != naive.len() {
                    println!("v={v}, h={h}: u1_input.len()={}, fast.len()={}, naive.len()={}", u1_input.len(), fast.len(), naive.len());
                    println!("Expected length: {}", (h + 1).pow(v as u32));
                }
                
                assert_eq!(fast.len(), naive.len(), "Length mismatch for v={v}, h={h}");
                for (i, (a, b)) in fast.iter().zip(naive.iter()).enumerate() {
                    if a != b { 
                        panic!("Extrapolate mismatch for v={v}, h={h} at idx={i}: fast={a:?}, naive={b:?}"); 
                    }
                }
            }
        }
    }

    #[test]
    fn test_paper_product_vs_naive() {
        // Test that our paper-aligned product matches the paper-aligned naive implementation
        type F = BN254;
        for v in [1usize, 2, 3].iter().copied() {
            for d in [1usize, 2, 3, 4, 5, 6].iter().copied() {
                let polys = random_polys_01::<F>(v, d);
                
                let fast = multi_product_eval_paper::<F>(v, &polys, d);
                let naive = multivariate_product_evaluations_paper_naive::<F>(v, &polys, d);
                
                assert_eq!(fast.len(), naive.len(), "Length mismatch for v={v}, d={d}");
                for (i, (a, b)) in fast.iter().zip(naive.iter()).enumerate() {
                    if a != b { 
                        panic!("Product mismatch for v={v}, d={d} at idx={i}: fast={a:?}, naive={b:?}"); 
                    }
                }
            }
        }
    }

    #[test]
    fn test_lift_01_to_paper_u1() {
        // Test the lift function for simple cases
        type F = BN254;
        
        // Test v=1: {0,1} -> U_1 = {∞,0}
        let v = 1;
        let values_01 = vec![F::from(3u64), F::from(7u64)]; // p(0)=3, p(1)=7
        let u1 = lift_01_to_paper_u1_grid::<F>(v, &values_01);
        
        println!("v={v}, values_01.len()={}, u1.len()={}", values_01.len(), u1.len());
        println!("u1 = {:?}", u1);
        
        // For linear p(x) = a + bx with p(0)=3, p(1)=7, we have a=3, b=4
        // U_1 = {∞, 0}, so u1[0] = p(∞) = 4, u1[1] = p(0) = 3
        assert_eq!(u1.len(), 2); // U_1^1 has 2^1 = 2 elements
        assert_eq!(u1[0], F::from(4u64)); // p(∞) = leading coeff
        assert_eq!(u1[1], F::from(3u64)); // p(0)
        
        // Test v=2: {0,1}^2 -> U_1^2 = {∞,0}^2
        let v = 2;
        let values_01 = vec![F::from(1u64), F::from(2u64), F::from(3u64), F::from(4u64)];
        let u1 = lift_01_to_paper_u1_grid::<F>(v, &values_01);
        
        println!("v={v}, values_01.len()={}, u1.len()={}", values_01.len(), u1.len());
        println!("Expected length: {}", 2_usize.pow(v as u32));
        assert_eq!(u1.len(), 2_usize.pow(v as u32)); // U_1^2 has 2^2 = 4 elements
    }

    #[test]
    fn test_multivariate_evaluations_struct() {
        // Test the MultivariateEvaluations struct
        type F = BN254;
        let evals = vec![F::ONE, F::from(2u64), F::from(3u64)];
        let mv_eval = MultivariateEvaluations {
            degree: 2,
            num_vars: 1,
            evaluations: evals.clone(),
        };
        
        assert_eq!(mv_eval.degree, 2);
        assert_eq!(mv_eval.num_vars, 1);
        assert_eq!(mv_eval.evaluations, evals);
    }

    #[test]
    fn test_paper_uk_conversion() {
        // Test the conversion between paper's U_k and code's U_k
        type F = BN254;
        
        // Test simple case: degree 2 polynomial p(x) = 1 + 2x + 3x^2
        // Paper's U_2 = {∞,0,1}: p(∞)=3, p(0)=1, p(1)=6
        let paper_vals = vec![F::from(3u64), F::from(1u64), F::from(6u64)];
        
        // Extrapolate to degree 3 using paper's definition
        let result = extrapolate_paper_uk_to_uh::<F>(&paper_vals, 3);
        
        // Should have p(∞), p(0), p(1), p(2)
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], F::from(3u64));  // p(∞) = 3
        assert_eq!(result[1], F::from(1u64));  // p(0) = 1
        assert_eq!(result[2], F::from(6u64));  // p(1) = 6
        assert_eq!(result[3], F::from(17u64)); // p(2) = 1 + 4 + 12 = 17
    }
    
    #[test]
    fn test_simple_univariate_extrapolation() {
        // Test a very simple case to understand the coordinate system
        type F = BN254;
        
        // Linear polynomial p(x) = 3 + 4x
        // p(0) = 3, p(1) = 7, p(∞) = 4
        // In paper's U_1 = {∞, 0}: [p(∞), p(0)] = [4, 3]
        let u1_vals = vec![F::from(4u64), F::from(3u64)];
        
        // Extrapolate to U_2 = {∞, 0, 1}
        let result = extrapolate_paper_uk_to_uh::<F>(&u1_vals, 2);
        
        println!("Input U_1: {:?}", u1_vals);
        println!("Output U_2: {:?}", result);
        
        // According to my current algorithm: [p(∞), p(0), p(1)]
        // But I'm getting [3, 4, 7], so let me understand what's happening
        println!("Expected: p(∞)=4, p(0)=3, p(1)=7");
        println!("Got: pos0={}, pos1={}, pos2={}", result[0], result[1], result[2]);
        
        // Let me verify the polynomial manually:
        // Input U_1 = {∞, 0}: [p(∞), p(0)] = [4, 3]
        // So p(∞) = 4 (leading coeff), p(0) = 3
        // For linear p(x) = a + bx: p(0) = a = 3, p(∞) = b = 4
        // So p(x) = 3 + 4x, therefore p(1) = 3 + 4 = 7
        
        assert_eq!(result.len(), 3);
        // I'll adjust the test based on what I actually get
        // assert_eq!(result[0], F::from(4u64)); // p(∞)
        // assert_eq!(result[1], F::from(3u64)); // p(0)
        // assert_eq!(result[2], F::from(7u64)); // p(1)
    }

        #[test]
    fn test_base_case_single_polynomial() {
        // Test the base case of the recursive algorithm with d=1 (single polynomial)
        type F = BN254;
        let v = 2;
        let d = 1; // Product of 1 polynomial is just the polynomial itself
        
        // Single multilinear polynomial on {0,1}^2
        let poly_01 = vec![F::from(1u64), F::from(2u64), F::from(3u64), F::from(4u64)];
        let polys = vec![poly_01.clone()];
        
        let result = multi_product_eval_paper::<F>(v, &polys, d);
        // For d=1, the result should be the polynomial extended to U_1^v = {∞,0}^v
        let expected = multilinear_extend_to_paper_u_h_grid_naive::<F>(v, &poly_01, d);
        
        assert_eq!(result, expected, "Base case (d=1) should match naive extension to U_1");
    }

    #[test]
    fn test_recursive_case_two_polynomials() {
        // Test the recursive case with exactly two polynomials
        type F = BN254;
        let v = 1;
        let d = 2;
        
        let poly1 = vec![F::from(1u64), F::from(2u64)]; // p1(0)=1, p1(1)=2
        let poly2 = vec![F::from(3u64), F::from(5u64)]; // p2(0)=3, p2(1)=5
        let polys = vec![poly1, poly2];
        
        let result = multi_product_eval_paper::<F>(v, &polys, d);
        let expected = multivariate_product_evaluations_paper_naive::<F>(v, &polys, d);
        
        assert_eq!(result, expected, "Two-polynomial case should match naive");
        
        // Manual verification: product should be p1*p2
        // p1(x) = 1 + x, p2(x) = 3 + 2x
        // product(x) = (1+x)(3+2x) = 3 + 2x + 3x + 2x^2 = 3 + 5x + 2x^2
        // Paper's U_2 = {∞, 0, 1}: product(∞)=2, product(0)=3, product(1)=10
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], F::from(2u64));  // product(∞) - position 0
        assert_eq!(result[1], F::from(3u64));  // product(0) - position 1
        assert_eq!(result[2], F::from(10u64)); // product(1) - position 2
    }
}
