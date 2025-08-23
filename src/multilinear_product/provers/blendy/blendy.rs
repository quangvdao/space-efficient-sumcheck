use crate::{
    hypercube::Hypercube,
    interpolation::LagrangePolynomial,
    messages::VerifierMessages,
    multilinear_product::TimeProductProver,
    order_strategy::{GraycodeOrder, SignificantBitOrder},
    streams::{Stream, StreamIterator},
};
use ark_ff::Field;
use ark_std::vec::Vec;
use std::collections::BTreeSet;

pub struct BlendyProductProver<F: Field, S: Stream<F>, const D: usize> {
    pub claim: F,
    pub current_round: usize,
    pub streams: [S; D],
    pub stream_iterators: [StreamIterator<F, S, SignificantBitOrder>; D],
    pub num_stages: usize,
    pub num_variables: usize,
    pub last_round_phase1: usize,
    pub verifier_messages: VerifierMessages<F>,
    pub verifier_messages_round_comp: VerifierMessages<F>,
    pub x_table: Vec<F>,
    pub y_table: Vec<F>,
    pub j_prime_table: Vec<Vec<F>>, // used when D==2
    pub partial_tables: Option<[Vec<F>; D]>, // used when D>2
    pub j_prime_table_flat: Option<Vec<F>>,   // used when D>2
    pub stage_size: usize,
    pub prev_table_round_num: usize,
    pub prev_table_size: usize,
    pub state_comp_set: BTreeSet<usize>,
    pub switched_to_vsbw: bool,
    pub vsbw_prover: TimeProductProver<F, S, D>,
}

impl<F: Field, S: Stream<F>, const D: usize> BlendyProductProver<F, S, D> {
    pub fn is_initial_round(&self) -> bool {
        self.current_round == 0
    }

    pub fn total_rounds(&self) -> usize {
        self.num_variables
    }

    pub fn init_round_vars(&mut self) {
        let n = self.num_variables;
        let j = self.current_round + 1;

        if let Some(&prev_round) = self.state_comp_set.range(..=j).next_back() {
            self.prev_table_round_num = prev_round;
            if let Some(&next_round) = self.state_comp_set.range((j + 1)..).next() {
                self.prev_table_size = next_round - prev_round;
            } else {
                self.prev_table_size = n + 1 - prev_round;
            }
        } else {
            self.prev_table_round_num = 0;
            self.prev_table_size = 0;
        }
    }

    pub fn compute_round(&mut self) -> Vec<F> {
        // Message shape: D==1 -> [1]; D>=2 -> [1, 2, 3, ..., D-1, ∞]
        if D == 1 {
            let mut sums: Vec<F> = vec![F::ZERO; 1];
            // switched to linear-time (vsbw) tail: accumulate only g(0)
            if self.switched_to_vsbw {
                // Calculate the bitmask for the number of free variables
                let bitmask: usize = 1 << (self.vsbw_prover.num_free_variables() - 1);
                // Determine the length of evaluations to iterate through
                let evaluations_len = match &self.vsbw_prover.evaluations[0] {
                    Some(evaluations) => evaluations.len(),
                    None => match &self.vsbw_prover.streams {
                        Some(streams) => 2usize.pow(streams[0].num_variables() as u32),
                        None => panic!("Both streams and evaluations cannot be None"),
                    },
                };
                for i in 0..(evaluations_len / 2) {
                    let v1 = match &self.vsbw_prover.evaluations[0] {
                        None => match &self.vsbw_prover.streams {
                            Some(streams) => streams[0].evaluation(i | bitmask),
                            None => panic!("Both streams and evaluations cannot be None"),
                        },
                        Some(evals) => evals[i | bitmask],
                    };
                    sums[0] += v1; // g(1)
                }
                return sums;
            }
            // streaming rounds (no tables) and general rounds for D==1: compute via lagrange over prefix
            // reset the streams
            self.stream_iterators.iter_mut().for_each(|stream_it| stream_it.reset());
            if self.is_initial_round() {
                for (_x_index, _) in Hypercube::<SignificantBitOrder>::new(self.num_variables - self.current_round - 1) {
                    let _v0 = self.stream_iterators[0].next().unwrap();
                    let v1 = self.stream_iterators[0].next().unwrap();
                    sums[0] += v1; // g(1)
                }
            } else {
                // Lagrange weights over current_round bits
                let mut sequential_lag_poly: LagrangePolynomial<F, SignificantBitOrder> =
                    LagrangePolynomial::new(&self.verifier_messages_round_comp);
                let lag_polys_len = Hypercube::<SignificantBitOrder>::stop_value(self.current_round);
                let mut lag_polys: Vec<F> = vec![F::ONE; lag_polys_len];
                for (x_index, _) in Hypercube::<SignificantBitOrder>::new(self.num_variables - self.current_round - 1) {
                    if x_index == 0 {
                        for (b_index, _) in Hypercube::<SignificantBitOrder>::new(self.current_round) {
                            lag_polys[b_index] = sequential_lag_poly.next().unwrap();
                        }
                    }
                    let mut partial_0: F = F::ZERO;
                    for (b_index, _) in Hypercube::<SignificantBitOrder>::new(self.current_round) {
                        let lag_poly = lag_polys[b_index];
                        partial_0 += self.stream_iterators[0].next().unwrap() * lag_poly;
                    }
                    let mut partial_1: F = F::ZERO;
                    for (b_index, _) in Hypercube::<SignificantBitOrder>::new(self.current_round) {
                        let lag_poly = lag_polys[b_index];
                        partial_1 += self.stream_iterators[0].next().unwrap() * lag_poly;
                    }
                    sums[0] += partial_1; // g(1)
                }
            }
            return sums;
        }

        let num_extras = if D > 2 { D - 2 } else { 0 };
        let mut sums: Vec<F> = vec![F::ZERO; if D > 1 { D } else { 1 }];

        // in the last rounds, we switch to the memory intensive prover
        if self.switched_to_vsbw {
            sums = self.vsbw_prover.vsbw_evaluate();
        }
        // if first few rounds, then no table is computed, need to compute sums from the streams
        else if self.current_round + 1 <= self.last_round_phase1 {
            // Lag Poly
            let mut sequential_lag_poly: LagrangePolynomial<F, SignificantBitOrder> =
                LagrangePolynomial::new(&self.verifier_messages_round_comp);
            let lag_polys_len = Hypercube::<SignificantBitOrder>::stop_value(self.current_round);
            let mut lag_polys: Vec<F> = vec![F::ONE; lag_polys_len];

            // reset the streams
            self.stream_iterators
                .iter_mut()
                .for_each(|stream_it| stream_it.reset());

            for (x_index, _) in
                Hypercube::<SignificantBitOrder>::new(self.num_variables - self.current_round - 1)
            {
                if self.is_initial_round() {
                    let mut g1 = F::ONE; let mut leading = F::ONE; let mut extras: Vec<F> = vec![F::ONE; num_extras];
                    for j in 0..D {
                        let v0 = self.stream_iterators[j].next().unwrap();
                        let v1 = self.stream_iterators[j].next().unwrap();
                        let diff = v1 - v0;
                        leading *= diff;
                        g1 *= v1;
                        if num_extras > 0 {
                            let mut val = v1 + diff; // z=2
                            extras[0] *= val;
                            for k in 1..num_extras { val += diff; extras[k] *= val; }
                        }
                    }
                    sums[0] += g1; for k in 0..num_extras { sums[1+k] += extras[k]; } sums[D-1] += leading;
                } else {
                    if x_index == 0 {
                        for (b_index, _) in Hypercube::<SignificantBitOrder>::new(self.current_round)
                        {
                            lag_polys[b_index] = sequential_lag_poly.next().unwrap();
                        }
                    }
                    let mut partial_0: [F; D] = [F::ZERO; D];
                    for (b_index, _) in Hypercube::<SignificantBitOrder>::new(self.current_round) {
                        let lag_poly = lag_polys[b_index];
                        for j in 0..D {
                            partial_0[j] += self.stream_iterators[j].next().unwrap() * lag_poly;
                        }
                    }
                    let mut partial_1: [F; D] = [F::ZERO; D];
                    for (b_index, _) in Hypercube::<SignificantBitOrder>::new(self.current_round) {
                        let lag_poly = lag_polys[b_index];
                        for j in 0..D {
                            partial_1[j] += self.stream_iterators[j].next().unwrap() * lag_poly;
                        }
                    }
                    let mut g1 = F::ONE; let mut leading = F::ONE; let mut extras: Vec<F> = vec![F::ONE; num_extras];
                    for j in 0..D {
                        let v1 = partial_1[j];
                        g1 *= v1;
                        let diff = v1 - partial_0[j];
                        leading *= diff;
                        if num_extras > 0 {
                            let mut val = v1 + diff; // z=2
                            extras[0] *= val;
                            for k in 1..num_extras { val += diff; extras[k] *= val; }
                        }
                    }
                    sums[0] += g1; for k in 0..num_extras { sums[1+k] += extras[k]; } sums[D-1] += leading;
                }
            }
            // no 1/2 scaling in ∞ scheme
        }
        // computing evaluations from the cross product tables
        else {
            if D == 2 {
                // things to help iterating
                let b_prime_num_vars = self.current_round + 1 - self.prev_table_round_num;
                let v_num_vars: usize =
                    self.prev_table_size + self.prev_table_round_num - self.current_round - 2;
                let b_prime_index_left_shift = v_num_vars + 1;

                // Lag Poly
                let mut sequential_lag_poly: LagrangePolynomial<F, GraycodeOrder> =
                    LagrangePolynomial::new(&self.verifier_messages_round_comp);
                let lag_polys_len = Hypercube::<GraycodeOrder>::stop_value(b_prime_num_vars);
                let mut lag_polys: Vec<F> = vec![F::ONE; lag_polys_len];

                // Sums
                for (b_prime_index, _) in Hypercube::<GraycodeOrder>::new(b_prime_num_vars) {
                    for (b_prime_prime_index, _) in Hypercube::<GraycodeOrder>::new(b_prime_num_vars)
                    {
                        // doing it like this, for each hypercube member lag_poly is computed exactly once
                        if b_prime_index == 0 {
                            lag_polys[b_prime_prime_index] = sequential_lag_poly.next().unwrap();
                        }

                        let lag_poly_1 = lag_polys[b_prime_index];
                        let lag_poly_2 = lag_polys[b_prime_prime_index];
                        let lag_poly = lag_poly_1 * lag_poly_2;
                        for (v_index, _) in Hypercube::<GraycodeOrder>::new(v_num_vars) {
                            let b_prime_0_v =
                                b_prime_index << b_prime_index_left_shift | 0 << v_num_vars | v_index;
                            let b_prime_prime_0_v = b_prime_prime_index << b_prime_index_left_shift
                                | 0 << v_num_vars
                                | v_index;
                            let b_prime_1_v =
                                b_prime_index << b_prime_index_left_shift | 1 << v_num_vars | v_index;
                            let b_prime_prime_1_v = b_prime_prime_index << b_prime_index_left_shift
                                | 1 << v_num_vars
                                | v_index;

                            // node 1 and ∞ via ± weights over the 4 corners
                            sums[0] += lag_poly * self.j_prime_table[b_prime_1_v][b_prime_prime_1_v];
                            // ∞: (+ +) − (+ −) − (− +) + (− −)
                            sums[D - 1] += lag_poly
                                * ( self.j_prime_table[b_prime_1_v][b_prime_prime_1_v]
                                    - self.j_prime_table[b_prime_1_v][b_prime_prime_0_v]
                                    - self.j_prime_table[b_prime_0_v][b_prime_prime_1_v]
                                    + self.j_prime_table[b_prime_0_v][b_prime_prime_0_v]);
                            // extras z ≥ 2
                            for (k, z_int) in (2..D).enumerate() {
                                let z = F::from(z_int as u32);
                                let mix = (F::ONE - z) * (F::ONE - z)
                                    * self.j_prime_table[b_prime_0_v][b_prime_prime_0_v]
                                    + (F::ONE - z) * z
                                        * self.j_prime_table[b_prime_0_v][b_prime_prime_1_v]
                                    + z * (F::ONE - z)
                                        * self.j_prime_table[b_prime_1_v][b_prime_prime_0_v]
                                    + z * z
                                        * self.j_prime_table[b_prime_1_v][b_prime_prime_1_v];
                                sums[1 + k] += lag_poly * mix;
                            }
                        }
                    }
                }
                // no 1/2 scaling
            } else {
                // D>2: consume from flat D-way cross product table
                let b_prime_num_vars = self.current_round + 1 - self.prev_table_round_num;
                let v_num_vars: usize =
                    self.prev_table_size + self.prev_table_round_num - self.current_round - 2;
                let shift = v_num_vars + 1;
                let side: usize = 1usize << (b_prime_num_vars + v_num_vars + 1);
                let table = self.j_prime_table_flat.as_ref().expect("flat table must exist");

                // Precompute lagrange weights over b_prime_num_vars for Graycode order
                let mut sequential_lag_poly: LagrangePolynomial<F, GraycodeOrder> =
                    LagrangePolynomial::new(&self.verifier_messages_round_comp);
                let lag_len = 1usize << b_prime_num_vars;
                let mut lag_vec: Vec<F> = vec![F::ONE; lag_len];
                for idx in 0..lag_len {
                    lag_vec[idx] = sequential_lag_poly.next().unwrap();
                }

                // Iterate b_prime indices for each dimension using mixed-radix counters
                let mut b_idxs = vec![0usize; D];
                loop {
                    // product of lagrange weights across dims
                    let mut lag_prod = F::ONE;
                    for j in 0..D { lag_prod *= lag_vec[b_idxs[j]]; }

                    // sum over v for node 1 directly
                    for v_index in 0..(1usize << v_num_vars) {
                        // node 0 and 1 fused indices per dim
                        let mut base0: Vec<usize> = Vec::with_capacity(D);
                        let mut base1: Vec<usize> = Vec::with_capacity(D);
                        for j in 0..D {
                            let fused0 = (b_idxs[j] << shift) | (0usize << v_num_vars) | v_index;
                            let fused1 = (b_idxs[j] << shift) | (1usize << v_num_vars) | v_index;
                            base0.push(fused0);
                            base1.push(fused1);
                        }
                        // flat index for tuple in row-major base 'side'
                        let mut idx1 = 0usize; let mut mul = 1usize;
                        for j in 0..D { idx1 += base1[j] * mul; mul *= side; }
                        sums[0] += lag_prod * table[idx1];

                        // node ∞: ± weights over s ∈ {0,1}^D
                        let mut s_mask = 0usize;
                        loop {
                            let mut sign = F::ONE; // product of (+1 for sj=1, −1 for sj=0)
                            let mut flat_idx = 0usize; let mut m = 1usize;
                            for j in 0..D {
                                let sj = (s_mask >> j) & 1;
                                sign *= if sj == 0 { -F::ONE } else { F::ONE };
                                let fused = (b_idxs[j] << shift) | (sj << v_num_vars) | v_index;
                                flat_idx += fused * m; m *= side;
                            }
                            sums[D - 1] += lag_prod * sign * table[flat_idx];
                            s_mask += 1; if s_mask >= (1usize << D) { break; }
                        }
                        // extras z ≥ 2: standard mixing with weights ∏_j ((1−z) if sj=0 else z)
                        for node_idx in 2..(2 + num_extras) {
                            let z = F::from(node_idx as u32);
                            let mut s_mask2 = 0usize;
                            loop {
                                let mut weight = F::ONE; let mut flat_idx2 = 0usize; let mut m2 = 1usize;
                                for j in 0..D {
                                    let sj = (s_mask2 >> j) & 1;
                                    weight *= if sj == 0 { F::ONE - z } else { z };
                                    let fused = (b_idxs[j] << shift) | (sj << v_num_vars) | v_index;
                                    flat_idx2 += fused * m2; m2 *= side;
                                }
                                sums[node_idx - 1] += lag_prod * weight * table[flat_idx2];
                                s_mask2 += 1; if s_mask2 >= (1usize << D) { break; }
                            }
                        }
                    }

                    // increment b_idxs in base 2^{b_prime_num_vars}
                    let mut k = 0usize;
                    while k < D {
                        b_idxs[k] += 1;
                        if b_idxs[k] < lag_len { break; }
                        b_idxs[k] = 0; k += 1;
                    }
                    if k == D { break; }
                }
            }
        }
        sums
    }

    pub fn compute_state(&mut self) {
        let j = self.current_round + 1;
        let p = self.state_comp_set.contains(&j);
        let is_largest = self.state_comp_set.range((j + 1)..).next().is_none();
        if p && !is_largest {
            // let time1 = std::time::Instant::now();
            let j_prime = self.prev_table_round_num;
            let t = self.prev_table_size;

            // println!(
            //     "table computation on round: {}, j_prime: {}, t: {}",
            //     j, j_prime, t
            // );

            // zero out the table(s)
            let table_len = Hypercube::<SignificantBitOrder>::stop_value(t);
            if D == 2 {
                self.j_prime_table = vec![vec![F::ZERO; table_len]; table_len];
            } else {
                // D-way flat tensor of size (2^t)^D
                let side = table_len;
                let total = side.pow(D as u32);
                self.j_prime_table_flat = Some(vec![F::ZERO; total]);
                // scratch partial tables per poly
                self.partial_tables = Some(std::array::from_fn(|_| vec![F::ZERO; side]));
            }

            // basically, this needs to get "zeroed" out at the beginning of state computation
            self.verifier_messages_round_comp = VerifierMessages::new_from_self(
                &self.verifier_messages,
                j_prime - 1,
                self.verifier_messages.messages.len(),
            );

            // some stuff for iterating
            let b_num_vars: usize = self.num_variables + 1 - j_prime - t;
            let x_num_vars = j_prime - 1;

            // Lag Poly
            let mut sequential_lag_poly: LagrangePolynomial<F, SignificantBitOrder> =
                LagrangePolynomial::new(&self.verifier_messages);

            assert!(x_num_vars == self.verifier_messages.messages.len());
            let lag_polys_len = Hypercube::<SignificantBitOrder>::stop_value(x_num_vars);
            let mut lag_polys: Vec<F> = vec![F::ONE; lag_polys_len];

            for (x_index, _) in Hypercube::<SignificantBitOrder>::new(x_num_vars) {
                lag_polys[x_index] = sequential_lag_poly.next().unwrap();
            }

            // reset the streams
            self.stream_iterators
                .iter_mut()
                .for_each(|stream_it| stream_it.reset());

            // Ensure x_table and y_table or partial tables are initialized
            if D == 2 {
                self.x_table = vec![F::ZERO; table_len];
                self.y_table = vec![F::ZERO; table_len];
            }

            for (_, _) in Hypercube::<SignificantBitOrder>::new(b_num_vars) {
                if D == 2 {
                    for (b_prime_index, _) in Hypercube::<SignificantBitOrder>::new(t) {
                        self.x_table[b_prime_index] = F::ZERO;
                        self.y_table[b_prime_index] = F::ZERO;

                        for (x_index, _) in Hypercube::<SignificantBitOrder>::new(x_num_vars) {
                            self.x_table[b_prime_index] +=
                                lag_polys[x_index] * self.stream_iterators[0].next().unwrap();
                            self.y_table[b_prime_index] +=
                                lag_polys[x_index] * self.stream_iterators[1].next().unwrap();
                        }
                    }
                    for (b_prime_index, _) in Hypercube::<SignificantBitOrder>::new(t) {
                        for (b_prime_prime_index, _) in Hypercube::<SignificantBitOrder>::new(t) {
                            self.j_prime_table[b_prime_index][b_prime_prime_index] +=
                                self.x_table[b_prime_index] * self.y_table[b_prime_prime_index];
                        }
                    }
                } else {
                    // D>2: fill partial tables per poly then take D-way outer product into flat tensor
                    let side = table_len;
                    let mut partials = self.partial_tables.take().unwrap();
                    for j in 0..D {
                        for (b_prime_index, _) in Hypercube::<SignificantBitOrder>::new(t) {
                            partials[j][b_prime_index] = F::ZERO;
                            for (x_index, _) in Hypercube::<SignificantBitOrder>::new(x_num_vars) {
                                partials[j][b_prime_index] +=
                                    lag_polys[x_index] * self.stream_iterators[j].next().unwrap();
                            }
                        }
                    }
                    // Outer product accumulation
                    let flat = self.j_prime_table_flat.as_mut().unwrap();
                    // Iterate over all tuples (i0,..,i_{D-1})
                    let mut idxs = vec![0usize; D];
                    loop {
                        let mut prod = F::ONE;
                        for j in 0..D { prod *= partials[j][idxs[j]]; }
                        // flat index: Σ idxs[j] * side^j
                        let mut flat_idx = 0usize;
                        let mut mult = 1usize;
                        for j in 0..D {
                            flat_idx += idxs[j] * mult;
                            mult *= side;
                        }
                        flat[flat_idx] += prod;
                        // increment idxs in mixed radix base 'side'
                        let mut k = 0usize;
                        while k < D {
                            idxs[k] += 1;
                            if idxs[k] < side { break; }
                            idxs[k] = 0; k += 1;
                        }
                        if k == D { break; }
                    }
                    self.partial_tables = Some(partials);
                }
            }
            // let time2 = std::time::Instant::now();
            // println!("table computation took: {:?}", time2 - time1);
        } else if p && is_largest {
            // switch to the memory intensive sumcheck on the last round computation
            let num_variables_new = self.num_variables - j + 1;
            self.switched_to_vsbw = true;

            // println!(
            //     "switched to vsbw on round: {}, num_vars_new: {}",
            //     j, num_variables_new
            // );

            // reset the streams
            self.stream_iterators
                .iter_mut()
                .for_each(|stream_it| stream_it.reset());

            // initialize the evaluations for the memory-intensive implementation for all D polynomials
            let side = 1 << num_variables_new;
            let mut evals: [Vec<F>; D] = std::array::from_fn(|_| vec![F::ZERO; side]);

            for (b_prime_index, _) in Hypercube::<SignificantBitOrder>::new(num_variables_new) {
                let mut sequential_lag_poly: LagrangePolynomial<F, SignificantBitOrder> =
                    LagrangePolynomial::new(&self.verifier_messages);
                for (_, _) in Hypercube::<SignificantBitOrder>::new(j - 1) {
                    let lag_poly = sequential_lag_poly.next().unwrap();
                    for t in 0..D {
                        evals[t][b_prime_index] +=
                            lag_poly * self.stream_iterators[t].next().unwrap();
                    }
                }
            }
            for t in 0..D {
                self.vsbw_prover.evaluations[t] = Some(std::mem::take(&mut evals[t]));
            }
        } else if self.switched_to_vsbw {
            let verifier_message = self.verifier_messages.messages[self.current_round - 1];
            self.vsbw_prover
                .vsbw_reduce_evaluations(verifier_message);
        }
    }
}
