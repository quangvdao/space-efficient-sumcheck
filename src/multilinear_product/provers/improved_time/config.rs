use ark_ff::Field;

use crate::{prover::ProductProverConfig, streams::Stream};

pub struct ImprovedTimeProductProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub num_variables: usize,
    pub claim: F,
    pub streams: Vec<S>,
}

impl<'a, F, S> ImprovedTimeProductProverConfig<F, S>
where
    F: Field,
    S: Stream<F>,
{
    pub fn new(claim: F, num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            claim,
            num_variables,
            streams,
        }
    }
}

impl<F: Field, S: Stream<F>> ProductProverConfig<F, S> for ImprovedTimeProductProverConfig<F, S> {
    fn default(claim: F, num_variables: usize, streams: Vec<S>) -> Self {
        Self {
            claim,
            num_variables,
            streams,
        }
    }
}
