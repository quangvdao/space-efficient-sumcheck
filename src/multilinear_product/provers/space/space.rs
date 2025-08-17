use ark_ff::Field;

use crate::{
	hypercube::Hypercube,
	interpolation::LagrangePolynomial,
	messages::VerifierMessages,
	order_strategy::SignificantBitOrder,
	streams::{Stream, StreamIterator},
};

pub struct SpaceProductProver<F: Field, S: Stream<F>, const D: usize> {
	pub claim: F,
	pub current_round: usize,
	pub stream_iterators: [StreamIterator<F, S, SignificantBitOrder>; D],
	pub num_variables: usize,
	pub verifier_messages: VerifierMessages<F>,
	pub inverse_four: F,
	pub inverse_two_pow_d: F,
}

impl<F: Field, S: Stream<F>, const D: usize> SpaceProductProver<F, S, D> {
	pub fn cty_evaluate(&mut self) -> Vec<F> {
		let mut nodes: Vec<F> = Vec::with_capacity(D + 1);
		nodes.push(F::ZERO);
		nodes.push(F::ONE);
		nodes.push(F::from(2_u32).inverse().unwrap());
		for k in 2..D { nodes.push(F::from(k as u32)); }
		let mut sums: Vec<F> = vec![F::ZERO; nodes.len()];

		// reset the streams
		self.stream_iterators.iter_mut().for_each(|it| it.reset());

		for (_, _) in Hypercube::<SignificantBitOrder>::new(self.num_variables - self.current_round - 1) {
			// can avoid unnecessary additions for first round since there is no lag poly: gives a small speedup
			if self.current_round == 0 {
				let mut prod_for_node: Vec<F> = vec![F::ONE; nodes.len()];
				for j in 0..D {
					let v0 = self.stream_iterators[j].next().unwrap();
					let v1 = self.stream_iterators[j].next().unwrap();
					prod_for_node[0] *= v0;
					prod_for_node[1] *= v1;
					prod_for_node[2] *= v0 + v1;
					for (idx, z) in nodes.iter().enumerate().skip(3) {
						let val = (F::ONE - *z) * v0 + *z * v1;
						prod_for_node[idx] *= val;
					}
				}
				for idx in 0..nodes.len() { sums[idx] += prod_for_node[idx]; }
			} else {
				let mut partial_0: [F; D] = [F::ZERO; D];
				let mut lag0: LagrangePolynomial<F, SignificantBitOrder> = LagrangePolynomial::new(&self.verifier_messages);
				for (_, _) in Hypercube::<SignificantBitOrder>::new(self.current_round) {
					let lp = lag0.next().unwrap();
					for j in 0..D {
						partial_0[j] += self.stream_iterators[j].next().unwrap() * lp;
					}
				}

				let mut partial_1: [F; D] = [F::ZERO; D];
				let mut lag1: LagrangePolynomial<F, SignificantBitOrder> = LagrangePolynomial::new(&self.verifier_messages);
				for (_, _) in Hypercube::<SignificantBitOrder>::new(self.current_round) {
					let lp = lag1.next().unwrap();
					for j in 0..D {
						partial_1[j] += self.stream_iterators[j].next().unwrap() * lp;
					}
				}

				let mut prod_for_node: Vec<F> = vec![F::ONE; nodes.len()];
				for j in 0..D {
					prod_for_node[0] *= partial_0[j];
					prod_for_node[1] *= partial_1[j];
					prod_for_node[2] *= partial_0[j] + partial_1[j];
					for (idx, z) in nodes.iter().enumerate().skip(3) {
						let val = (F::ONE - *z) * partial_0[j] + *z * partial_1[j];
						prod_for_node[idx] *= val;
					}
				}
				for idx in 0..nodes.len() { sums[idx] += prod_for_node[idx]; }
			}
		}
		// scale 1/2 node
		if nodes.len() > 2 { sums[2] = sums[2] * self.inverse_two_pow_d; }
		sums
	}
}
