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
}

impl<F: Field, S: Stream<F>, const D: usize> SpaceProductProver<F, S, D> {
	pub fn cty_evaluate(&mut self) -> Vec<F> {
		let num_extras = if D > 2 { D - 2 } else { 0 };
		let mut sums: Vec<F> = vec![F::ZERO; 2 + num_extras];

		// reset the streams
		self.stream_iterators.iter_mut().for_each(|it| it.reset());

		for (_, _) in Hypercube::<SignificantBitOrder>::new(self.num_variables - self.current_round - 1) {
			// can avoid unnecessary additions for first round since there is no lag poly: gives a small speedup
			if self.current_round == 0 {
				let mut g0 = F::ONE;
				let mut leading = F::ONE;
				let mut extras: Vec<F> = vec![F::ONE; num_extras];
				for j in 0..D {
					let v0 = self.stream_iterators[j].next().unwrap();
					let v1 = self.stream_iterators[j].next().unwrap();
					let diff = v1 - v0;
					g0 *= v0;
					leading *= diff;
					if num_extras > 0 {
						let v1 = v0 + diff;
						let mut val = v1 + diff; // z=2
						extras[0] *= val;
						for k in 1..num_extras { val += diff; extras[k] *= val; }
					}
				}
				sums[0] += g0; sums[1] += leading; for k in 0..num_extras { sums[2+k] += extras[k]; }
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

				let mut g0 = F::ONE; let mut leading = F::ONE; let mut extras: Vec<F> = vec![F::ONE; num_extras];
				for j in 0..D {
					let v0 = partial_0[j];
					let diff = partial_1[j] - partial_0[j];
					g0 *= v0;
					leading *= diff;
					if num_extras > 0 {
						let v1 = v0 + diff;
						let mut val = v1 + diff; // z=2
						extras[0] *= val;
						for k in 1..num_extras { val += diff; extras[k] *= val; }
					}
				}
				sums[0] += g0; sums[1] += leading; for k in 0..num_extras { sums[2+k] += extras[k]; }
			}
		}
		sums
	}
}
