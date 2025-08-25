use ark_ff::Field;

use crate::{prover::ProductProverConfig, streams::Stream};

pub struct TimeEqProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_variables: usize,
    pub claim: F,
    pub streams: Vec<S>,
    pub w: Vec<F>,
}

impl<'a, F, S> TimeEqProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(claim: F, num_variables: usize, streams: Vec<S>, w: Vec<F>) -> Self {
        Self {
            claim,
            num_variables,
            streams,
            w,
        }
    }
}

impl<F: Field, S: Stream<F>> ProductProverConfig<F, S> for TimeEqProverConfig<F, S> {
    fn default(claim: F, num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            claim,
            num_variables,
            streams,
            w: vec![F::ONE; num_variables],
        }
    }
}
