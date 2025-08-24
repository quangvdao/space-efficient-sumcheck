use ark_ff::Field;
use crate::order_strategy::GraycodeOrder;
use crate::interpolation::LagrangePolynomial;

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

/// Grid node: either a finite field element or an evaluation at infinity (leading coefficient).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node<F> {
    Finite(F),
    Infinity,
}

/// Trait to allow pluggable univariate extrapolation implementations.
pub trait UniExtrap<F: Field> {
    fn extrapolate(x_nodes: &[F], x_values: &[F], y_nodes: &[F]) -> Vec<F>;
}

/// Default extrapolator using barycentric Lagrange interpolation.
pub struct BarycentricExtrap;

impl<F: Field> UniExtrap<F> for BarycentricExtrap {
    fn extrapolate(x_nodes: &[F], x_values: &[F], y_nodes: &[F]) -> Vec<F> {
        univariate_extrapolate::<F>(x_nodes, x_values, y_nodes)
    }
}

/// Multivariate extrapolation: given evaluations on a grid `U_k^v` specified by per-axis nodes
/// `axes_k[j]` (length k+1 each), extend to evaluations on `U_d^v` given `axes_d[j]`.
/// The input `values` is a flat array in row-major order with last axis contiguous.
pub fn multivariate_extrapolate<F: Field>(
    v: usize,
    axes_k: &[Vec<F>],
    axes_d: &[Vec<F>],
    values: &[F],
) -> Vec<F> {
    multivariate_extrapolate_with::<F, BarycentricExtrap>(v, axes_k, axes_d, values)
}

/// Generic version of multivariate extrapolation using a provided univariate extrapolation strategy.
pub fn multivariate_extrapolate_with<F: Field, E: UniExtrap<F>>(
    v: usize,
    axes_k: &[Vec<F>],
    axes_d: &[Vec<F>],
    values: &[F],
) -> Vec<F> {
    assert_eq!(axes_k.len(), v);
    assert_eq!(axes_d.len(), v);
    for j in 0..v { assert!(axes_k[j].len() >= 2 && axes_d[j].len() >= axes_k[j].len()); }

    // Start with the input grid
    let mut cur = values.to_vec();
    let mut cur_axes: Vec<Vec<F>> = axes_k.to_vec();

    // Helper to compute strides for a given axis shape
    let mut shape: Vec<usize> = cur_axes.iter().map(|ax| ax.len()).collect();

    for axis in 0..v {
        let src_nodes = &cur_axes[axis];
        let dst_nodes = &axes_d[axis];
        if dst_nodes.len() == src_nodes.len() { continue; }

        // Compute strides for current and next (after expansion along `axis`)
        let mut strides_src = vec![1usize; v];
        for i in (0..v-1).rev() { strides_src[i] = strides_src[i+1] * shape[i+1]; }
        let mut next_shape = shape.clone();
        next_shape[axis] = dst_nodes.len();
        let mut strides_dst = vec![1usize; v];
        for i in (0..v-1).rev() { strides_dst[i] = strides_dst[i+1] * next_shape[i+1]; }

        // Number of independent slices for this axis (product of sizes excluding this axis)
        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() { if i != axis { num_slices *= *s; } }

        let src_axis_len = shape[axis];
        let dst_axis_len = dst_nodes.len();

        // Total elements in next grid
        let total_next_elems = next_shape.iter().copied().fold(1usize, |acc, s| acc * s);
        let mut next = vec![F::ZERO; total_next_elems];

        for slice_idx in 0..num_slices {
            // Derive coordinates for axes != axis from slice_idx
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

            // Gather univariate line along the axis from src
            let mut line: Vec<F> = Vec::with_capacity(src_axis_len);
            for t in 0..src_axis_len {
                line.push(cur[src_base + t * strides_src[axis]]);
            }
            // Extrapolate this line
            let line_ex = E::extrapolate(src_nodes, &line, dst_nodes);
            // Scatter into destination grid along the axis
            for t in 0..dst_axis_len {
                next[dst_base + t * strides_dst[axis]] = line_ex[t];
            }
        }

        // Update current grid and shape for next axis
        cur = next;
        cur_axes[axis] = dst_nodes.clone();
        shape = next_shape;
    }

    cur
}

/// Compute s(∞) (leading coefficient) from finite nodes and values using Lagrange weights.
fn compute_leading_coeff_from_finite<F: Field>(finite_nodes: &[F], finite_values: &[F]) -> F {
    let k = finite_nodes.len();
    assert!(k >= 1);
    // weights w_i = 1 / ∏_{j≠i} (x_i - x_j)
    let mut acc = F::from(0u32);
    for i in 0..k {
        let xi = finite_nodes[i];
        let mut denom = F::from(1u32);
        for j in 0..k {
            if i == j { continue; }
            denom *= xi - finite_nodes[j];
        }
        let wi = denom.inverse().expect("distinct nodes required");
        acc += finite_values[i] * wi;
    }
    acc
}

/// Univariate extrapolation that supports node sets with infinity (as leading coefficient).
pub fn univariate_extrapolate_nodes<F: Field>(
    x_nodes: &[Node<F>],
    x_values: &[F],
    y_nodes: &[Node<F>],
) -> Vec<F> {
    assert_eq!(x_nodes.len(), x_values.len());
    // Separate finite nodes and optional s_inf source
    let mut finite_nodes: Vec<F> = Vec::new();
    let mut finite_values: Vec<F> = Vec::new();
    let mut s_inf_src: Option<F> = None;
    for (n, v) in x_nodes.iter().copied().zip(x_values.iter().copied()) {
        match n {
            Node::Finite(x) => { finite_nodes.push(x); finite_values.push(v); }
            Node::Infinity => { s_inf_src = Some(v); }
        }
    }
    let k_src = finite_nodes.len();
    let has_inf_src = x_nodes.iter().any(|n| matches!(n, Node::Infinity));
    let deg_src = if has_inf_src { k_src } else { k_src.saturating_sub(1) };
    // If no s_inf provided, compute it from finite nodes
    let s_inf = s_inf_src.unwrap_or_else(|| compute_leading_coeff_from_finite::<F>(&finite_nodes, &finite_values));

    // Helper to evaluate at a single finite r depending on available data
    let eval_at = |r: F| -> F {
        if has_inf_src {
            match k_src {
                0 => F::from(0u32),
                1 => {
                    let x0 = finite_nodes[0];
                    let v0 = finite_values[0];
                    s_inf * (r - x0) + v0
                }
                _ => LagrangePolynomial::<F, GraycodeOrder>::evaluate_from_infty_and_points(
                    r,
                    s_inf,
                    &finite_nodes,
                    &finite_values,
                ),
            }
        } else {
            // Pure barycentric from finite nodes only
            univariate_extrapolate::<F>(&finite_nodes, &finite_values, core::slice::from_ref(&r))[0]
        }
    };
    // Evaluate for each y
    let mut out: Vec<F> = Vec::with_capacity(y_nodes.len());
    // Target degree from y_nodes: count finite nodes before potential Infinity
    let mut deg_tgt = 0usize;
    for n in y_nodes.iter() { if let Node::Finite(_) = n { deg_tgt += 1; } }
    for &y in y_nodes.iter() {
        match y {
            Node::Finite(r) => { out.push(eval_at(r)); }
            Node::Infinity => {
                let val = if deg_tgt > deg_src { F::from(0u32) } else { s_inf };
                out.push(val);
            }
        }
    }
    out
}

/// Multivariate extrapolation using node sets that may include Infinity at the end of each axis.
pub fn multivariate_extrapolate_nodes<F: Field>(
    v: usize,
    axes_k: &[Vec<Node<F>>],
    axes_d: &[Vec<Node<F>>],
    values: &[F],
) -> Vec<F> {
    assert_eq!(axes_k.len(), v);
    assert_eq!(axes_d.len(), v);
    for j in 0..v { assert!(axes_k[j].len() >= 2 && axes_d[j].len() >= axes_k[j].len()); }

    let mut cur = values.to_vec();
    let mut cur_nodes: Vec<Vec<Node<F>>> = axes_k.to_vec();
    let mut shape: Vec<usize> = cur_nodes.iter().map(|ax| ax.len()).collect();

    for axis in 0..v {
        let src_nodes = &cur_nodes[axis];
        let dst_nodes = &axes_d[axis];
        if dst_nodes.len() == src_nodes.len() { continue; }

        let mut strides_src = vec![1usize; v];
        for i in (0..v-1).rev() { strides_src[i] = strides_src[i+1] * shape[i+1]; }
        let mut next_shape = shape.clone();
        next_shape[axis] = dst_nodes.len();
        let mut strides_dst = vec![1usize; v];
        for i in (0..v-1).rev() { strides_dst[i] = strides_dst[i+1] * next_shape[i+1]; }

        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() { if i != axis { num_slices *= *s; } }

        let src_axis_len = shape[axis];
        let dst_axis_len = dst_nodes.len();
        let total_next_elems = next_shape.iter().copied().fold(1usize, |acc, s| acc * s);
        let mut next = vec![F::from(0u32); total_next_elems];

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
            let line_ex = univariate_extrapolate_nodes::<F>(src_nodes, &line, dst_nodes);
            for t in 0..dst_axis_len { next[dst_base + t * strides_dst[axis]] = line_ex[t]; }
        }

        cur = next;
        cur_nodes[axis] = dst_nodes.clone();
        shape = next_shape;
    }

    cur
}

/// Compute product of d multilinear polynomials in evaluation form on {0,1}^v,
/// returning evaluations of the product on U_d^v. API follows the paper's Algorithm 1.
/// - inputs: list of polynomials, each as a flat vector of length 2^v, ordered with last axis contiguous.
/// - axes_d: target per-axis nodes for U_d (length d+1 each axis, identical lists per axis typically).
pub fn multivariate_product_evaluations<F: Field>(
    v: usize,
    polys_01: &[Vec<F>],
    axes_d: &[Vec<F>],
) -> Vec<F> {
    assert!(v >= 1);
    assert!(!polys_01.is_empty());
    assert_eq!(axes_d.len(), v);
    let expected_len = 1usize << v;
    for p in polys_01 { assert_eq!(p.len(), expected_len, "each polynomial must have 2^v evaluations"); }

    // Recursive procedure matching Algorithm 1: at each level, evaluate halves on their own grids,
    // extrapolate both to the parent grid, and multiply pointwise.
    fn rec<F: Field>(
        v: usize,
        polys: &[Vec<F>],
        axes_parent: &[Vec<F>],
    ) -> Vec<F> {
        let d_here = polys.len();
        if d_here == 1 {
            // Base: already given on U_1^v == {0,1}^v; if parent also has degree 1, extrapolate is identity
            let axes_01: Vec<Vec<F>> = (0..v).map(|_| vec![F::ZERO, F::ONE]).collect();
            return multivariate_extrapolate::<F>(v, &axes_01, axes_parent, &polys[0]);
        }
        let m = d_here / 2;
        let axes_left = (0..v).map(|_| (1..=m).map(|x| F::from(x as u32)).collect::<Vec<F>>()).collect::<Vec<Vec<F>>>();
        let axes_right = (0..v).map(|_| (1..=(d_here-m)).map(|x| F::from(x as u32)).collect::<Vec<F>>()).collect::<Vec<Vec<F>>>();
        let eval_left_small = rec::<F>(v, &polys[..m], &axes_left);
        let eval_right_small = rec::<F>(v, &polys[m..], &axes_right);
        // Extrapolate both to parent axes
        let eval_left = multivariate_extrapolate::<F>(v, &axes_left, axes_parent, &eval_left_small);
        let eval_right = multivariate_extrapolate::<F>(v, &axes_right, axes_parent, &eval_right_small);
        // Pointwise product on parent grid
        let mut out = eval_left;
        for (o, r) in out.iter_mut().zip(eval_right.iter()) { *o *= *r; }
        out
    }

    rec::<F>(v, polys_01, axes_d)
}

/// Same as multivariate_product_evaluations, but with node sets {1,2,...,D, ∞} per axis.
/// Callers supply only D; this function constructs the nodes and computes the product.
pub fn multivariate_product_evaluations_nodes<F: Field>(
    v: usize,
    polys_01: &[Vec<F>],
    d: usize,
) -> Vec<F> {
    assert!(v >= 1);
    assert!(!polys_01.is_empty());
    let expected_len = 1usize << v;
    for p in polys_01 { assert_eq!(p.len(), expected_len, "each polynomial must have 2^v evaluations"); }

    let make_nodes = |deg: usize| -> Vec<Node<F>> {
        let mut nodes = (1..=deg).map(|x| Node::Finite(F::from(x as u32))).collect::<Vec<_>>();
        nodes.push(Node::Infinity);
        nodes
    };
    let _ = (0..v).map(|_| make_nodes(d)).collect::<Vec<_>>();

    // Recursive function working with node sets
    fn rec<F: Field>(
        v: usize,
        polys: &[Vec<F>],
        deg_parent: usize,
    ) -> Vec<F> {
        if polys.len() == 1 {
            let axes_k: Vec<Vec<Node<F>>> = (0..v).map(|_| vec![Node::Finite(F::from(0u32)), Node::Finite(F::from(1u32))]).collect();
            let axes_d: Vec<Vec<Node<F>>> = (0..v).map(|_| {
                let mut n = (1..=deg_parent).map(|x| Node::Finite(F::from(x as u32))).collect::<Vec<_>>();
                n.push(Node::Infinity);
                n
            }).collect();
            return multivariate_extrapolate_nodes::<F>(v, &axes_k, &axes_d, &polys[0]);
        }
        let d_here = polys.len();
        let m = d_here / 2;
        let left = rec::<F>(v, &polys[..m], m);
        let right = rec::<F>(v, &polys[m..], d_here - m);
        let axes_left: Vec<Vec<Node<F>>> = (0..v).map(|_| {
            let mut n = (1..=m).map(|x| Node::Finite(F::from(x as u32))).collect::<Vec<_>>();
            n.push(Node::Infinity);
            n
        }).collect();
        let axes_right: Vec<Vec<Node<F>>> = (0..v).map(|_| {
            let mut n = (1..=(d_here - m)).map(|x| Node::Finite(F::from(x as u32))).collect::<Vec<_>>();
            n.push(Node::Infinity);
            n
        }).collect();
        let axes_parent: Vec<Vec<Node<F>>> = (0..v).map(|_| {
            let mut n = (1..=deg_parent).map(|x| Node::Finite(F::from(x as u32))).collect::<Vec<_>>();
            n.push(Node::Infinity);
            n
        }).collect();
        let left_ex = multivariate_extrapolate_nodes::<F>(v, &axes_left, &axes_parent, &left);
        let right_ex = multivariate_extrapolate_nodes::<F>(v, &axes_right, &axes_parent, &right);
        let mut out = left_ex;
        for (o, r) in out.iter_mut().zip(right_ex.iter()) { *o *= *r; }
        out
    }

    rec::<F>(v, polys_01, d)
}

/// Generalized version: accepts explicit per-axis node sets for the target grid.
pub fn multivariate_product_evaluations_nodes_with_axes<F: Field>(
    v: usize,
    polys_01: &[Vec<F>],
    axes_d: &[Vec<Node<F>>],
) -> Vec<F> {
    assert!(v >= 1);
    assert!(!polys_01.is_empty());
    assert_eq!(axes_d.len(), v);
    let expected_len = 1usize << v;
    for p in polys_01 { assert_eq!(p.len(), expected_len, "each polynomial must have 2^v evaluations"); }

    fn rec<F: Field>(
        v: usize,
        polys: &[Vec<F>],
        axes_parent: &[Vec<Node<F>>],
    ) -> Vec<F> {
        let d_here = polys.len();
        if d_here == 1 {
            let axes_k: Vec<Vec<Node<F>>> = (0..v)
                .map(|_| vec![Node::Finite(F::from(0u32)), Node::Finite(F::from(1u32))])
                .collect();
            return multivariate_extrapolate_nodes::<F>(v, &axes_k, axes_parent, &polys[0]);
        }
        let m = d_here / 2;
        let left = rec::<F>(v, &polys[..m], axes_parent);
        let right = rec::<F>(v, &polys[m..], axes_parent);
        let mut out = left;
        for (o, r) in out.iter_mut().zip(right.iter()) { *o *= *r; }
        out
    }

    rec::<F>(v, polys_01, axes_d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::F19 as Fp;
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_univariate_extrapolate_identity() {
        let x: Vec<Fp> = vec![Fp::from(0u32), Fp::from(1u32), Fp::from(2u32)];
        let vals: Vec<Fp> = vec![Fp::from(3u32), Fp::from(5u32), Fp::from(7u32)];
        let y = x.clone();
        let out = univariate_extrapolate(&x, &vals, &y);
        assert_eq!(out, vals);
    }

    fn eval_multilinear(v: usize, values_01: &[Fp], point: &[Fp]) -> Fp {
        // Evaluate multilinear polynomial via tensor-product Lagrange on {0,1}^v
        assert_eq!(values_01.len(), 1usize << v);
        let mut acc = Fp::from(0u32);
        for idx in 0..(1usize << v) {
            // weight = Π_j (x_j if bit=1 else 1-x_j)
            let mut w = Fp::from(1u32);
            for j in 0..v {
                let bit = (idx >> j) & 1;
                if bit == 1 { w *= point[j]; } else { w *= Fp::from(1u32) - point[j]; }
            }
            acc += w * values_01[idx];
        }
        acc
    }

    #[test]
    fn test_multivariate_extrapolate_matches_direct_eval_small_nodes_1_to_d_inf() {
        // v=2, start at {0,1}^2, target axes_d with nodes [1,2,3, ∞]
        let v = 2usize;
        let mut rng = StdRng::seed_from_u64(42);
        let values_01: Vec<Fp> = (0..(1<<v)).map(|_| Fp::from(rng.gen::<u64>())).collect();
        // axes
        let axes_k: Vec<Vec<Node<Fp>>> = (0..v).map(|_| vec![Node::Finite(Fp::from(0u32)), Node::Finite(Fp::from(1u32))]).collect();
        let axes_d: Vec<Vec<Node<Fp>>> = (0..v).map(|_| vec![Node::Finite(Fp::from(1u32)), Node::Finite(Fp::from(2u32)), Node::Finite(Fp::from(3u32)), Node::Infinity]).collect();
        // Extrapolate grid
        let grid = multivariate_extrapolate_nodes::<Fp>(v, &axes_k, &axes_d, &values_01);
        // Check finite grid points against direct evaluation; skip any with ∞ on any axis
        for i in 0..axes_d[0].len() {
            for j in 0..axes_d[1].len() {
                match (axes_d[0][i], axes_d[1][j]) {
                    (Node::Finite(x0), Node::Finite(x1)) => {
                        let point = [x0, x1];
                        let expected = eval_multilinear(v, &values_01, &point);
                        let idx = i * axes_d[1].len() + j;
                        assert_eq!(grid[idx], expected);
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn test_multivariate_product_vs_baseline_nodes_1_to_d_inf() {
        // v=2, d=3, axes_d = [1,2,3, ∞] per axis
        let v = 2usize;
        let d = 3usize;
        let mut rng = StdRng::seed_from_u64(7);
        let polys_01: Vec<Vec<Fp>> = (0..d)
            .map(|_| (0..(1<<v)).map(|_| Fp::from(rng.gen::<u64>())).collect())
            .collect();
        let axes_d: Vec<Vec<Node<Fp>>> = (0..v).map(|_| vec![Node::Finite(Fp::from(1u32)), Node::Finite(Fp::from(2u32)), Node::Finite(Fp::from(3u32)), Node::Infinity]).collect();

        // Baseline: extrapolate each to node grid, then multiply pointwise
        let axes_01: Vec<Vec<Node<Fp>>> = (0..v).map(|_| vec![Node::Finite(Fp::from(0u32)), Node::Finite(Fp::from(1u32))]).collect();
        let mut baseline = vec![Fp::from(1u32); (d+1).pow(v as u32) as usize];
        for p in polys_01.iter() {
            let ex = multivariate_extrapolate_nodes::<Fp>(v, &axes_01, &axes_d, p);
            for (b, e) in baseline.iter_mut().zip(ex.iter()) { *b *= *e; }
        }

        // Our recursive algorithm
        let ours = multivariate_product_evaluations_nodes::<Fp>(v, &polys_01, d);
        assert_eq!(ours, baseline);
    }
}


