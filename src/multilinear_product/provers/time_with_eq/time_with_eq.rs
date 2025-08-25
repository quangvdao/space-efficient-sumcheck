use ark_ff::Field;
use ark_std::vec::Vec;

use crate::{
    interpolation::{
        equality_polynomial::{eq_linear_on_u_d, eq_split_eval_at, eq_split_tables},
    },
    streams::Stream,
};

/// TimeProductProverWithEq implements the linear-time (VSBW) product prover specialized to
/// g(X) = eq(w, X) * \prod_k p_k(X), aligning with the paper's "Equality Polynomial Optimization".
///
/// High-level alignment with the LaTeX (Section: Equality Polynomial Optimization):
/// - Decomposition: eq(w, X) factors as prefix * window * suffix. In a 1-round (linear-time) flow,
///   the "window" is just the current axis i: eq(w_i, X_i) is linear, while the suffix is over
///   remaining free variables. The prefix term eq(w_{<i}, r_{<i]}) is a scalar multiplier (alpha).
/// - Suffix handling: k=2 split over remaining bits using DP to achieve square-root memory.
/// - Message nodes U_D: we evaluate the window linear factor at U_D via eq_linear_on_u_d.
pub struct TimeProductProverWithEq<F: Field, S: Stream<F>, const D: usize> {
    pub claim: F,
    pub current_round: usize,
    pub evaluations: [Option<Vec<F>>; D],
    pub streams: Option<[S; D]>,
    pub num_variables: usize,
    pub w: Vec<F>, // equality vector length = num_variables
    pub prefix_alpha: F, // \prod_{t < current_round} ell_t(r_t)
}

impl<'a, F: Field, S: Stream<F>, const D: usize> TimeProductProverWithEq<F, S, D> {
    pub fn total_rounds(&self) -> usize { self.num_variables }
    pub fn num_free_variables(&self) -> usize { self.num_variables - self.current_round }

    fn compute_ti_evaluations(&self) -> Vec<F> {
        let num_extras = if D > 1 { D - 1 } else { 0 };
        let mut sums: Vec<F> = vec![F::ZERO; D + 1];
        let bitmask: usize = 1 << (self.num_free_variables() - 1);

        let s_rest = if self.num_free_variables() > 0 {
            self.num_free_variables() - 1
        } else {
            0
        };
        let (left, right, left_mask, right_shift) = if s_rest == 0 {
            (Vec::<F>::new(), Vec::<F>::new(), 0usize, 0usize)
        } else {
            let (l, r) = eq_split_tables::<F>(
                &self.w[(self.current_round + 1)..(self.current_round + 1 + s_rest)],
            );
            let s_left = s_rest / 2;
            (l, r, (1usize << s_left) - 1, s_left)
        };

        let process_chunk = |i: usize, sums: &mut Vec<F>| {
            let eq_suffix = if s_rest == 0 {
                F::ONE
            } else {
                let idx_l = i & left_mask;
                let idx_r = i >> right_shift;
                eq_split_eval_at::<F>(&left, &right, idx_l, idx_r)
            };

            let mut g0 = F::ONE;
            let mut g1 = F::ONE;
            let mut g_inf = F::ONE;
            let mut extras = [F::ONE; 30];

            for j in 0..D {
                let (v0, v1) = if self.current_round == 0 {
                    let streams = self.streams.as_ref().expect("streams");
                    (streams[j].evaluation(i), streams[j].evaluation(i | bitmask))
                } else {
                    let tables: [&[F]; D] = std::array::from_fn(|k| self.evaluations[k].as_ref().unwrap().as_slice());
                    (tables[j][i], tables[j][i | bitmask])
                };

                let diff = v1 - v0;
                g0 *= v0;
                g1 *= v1;
                g_inf *= diff;
                if num_extras > 0 {
                    let mut val = v1 + diff;
                    extras[0] *= val;
                    for k in 1..num_extras {
                        val += diff;
                        extras[k] *= val;
                    }
                }
            }
            // Accumulate weighted products
            sums[0] += eq_suffix * g0; // z=0
            sums[1] += eq_suffix * g1; // z=1
            for k in 0..num_extras -1 { // z=2..D-1
                sums[2 + k] += eq_suffix * extras[k];
            }
            if D > 1 {
                sums[D] += eq_suffix * g_inf; // z=inf
            }
        };

        let evaluations_len = if self.current_round == 0 {
            1usize << self.streams.as_ref().unwrap()[0].num_variables()
        } else {
            self.evaluations[0].as_ref().unwrap().len()
        };

        for i in 0..(evaluations_len / 2) {
            process_chunk(i, &mut sums);
        }

        sums
    }

    pub fn evaluate(&self) -> Vec<F> {
        let ti_evals = self.compute_ti_evaluations();
        let wi = self.w[self.current_round];
        let (linear_vals_at_u_d, slope_inf) = eq_linear_on_u_d::<F>(wi, D + 1);
        let linear_const = F::ONE - wi;
        let l_at_0 = linear_const;
        let l_at_1 = F::ONE - wi + (wi + wi - F::ONE);

        let ti_at_0 = ti_evals[0];
        let ti_at_1_check = (self.claim - l_at_0 * ti_at_0) * l_at_1.inverse().unwrap();
        
        let ti_at_1 = ti_evals[1];
        assert_eq!(ti_at_1, ti_at_1_check, "Sumcheck consistency failed");

        let mut out = vec![F::ZERO; D + 2];
        out[0] = l_at_0 * ti_at_0; // s(0)
        out[1] = l_at_1 * ti_at_1; // s(1)

        for i in 0..D-1 { // s(2)...s(D)
            out[2+i] = linear_vals_at_u_d[i] * ti_evals[2+i];
        }
        let scale_inf = if slope_inf.is_zero() { linear_const } else { slope_inf };
        out[D+1] = scale_inf * ti_evals[D]; // s(inf)
        
        if !self.prefix_alpha.is_one() {
            for v in out.iter_mut() {
                *v *= self.prefix_alpha;
            }
        }
        out
    }

    pub fn reduce_evaluations(&mut self, verifier_message: F) {
        for i in 0..D {
            let setbit: usize = 1 << self.num_free_variables();
            if let Some(mut prev) = self.evaluations[i].take() {
                let evaluations_len = prev.len() / 2;
                for i0 in 0..evaluations_len {
                    let i1 = i0 | setbit;
                    let e0 = prev[i0];
                    let e1 = prev[i1];
                    let diff = e1 - e0;
                    prev[i0] = e0 + diff * verifier_message;
                }
                prev.truncate(evaluations_len);
                self.evaluations[i] = Some(prev);
            } else {
                let evaluations_len = 1usize << self.num_free_variables();
                let streams = self.streams.as_ref().expect("Both streams and evaluations cannot be None");
                let mut out: Vec<F> = vec![F::ZERO; evaluations_len];
                for i0 in 0..evaluations_len {
                    let i1 = i0 | setbit;
                    let e0 = streams[i].evaluation(i0);
                    let e1 = streams[i].evaluation(i1);
                    let diff = e1 - e0;
                    out[i0] = e0 + diff * verifier_message;
                }
                self.evaluations[i] = Some(out);
            }
        }
        // Update prefix alpha with the bound variable's linear factor at r: ell(r) = (1-w) + (2w-1) r
        let idx_prev = self.current_round;
        let w_prev = self.w[idx_prev];
        let slope = w_prev + w_prev - F::ONE; // 2w-1
        let ell_at_r = (F::ONE - w_prev) + slope * verifier_message;
        self.prefix_alpha *= ell_at_r;
    }
}
