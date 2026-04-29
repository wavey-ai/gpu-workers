use std::collections::BTreeMap;

use anyhow::Result;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JobContext {
    pub seq: Option<u32>,
    pub chunk_id: Option<u64>,
    pub stream_id: Option<u64>,
    pub labels: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct JobEnvelope<T> {
    pub id: u64,
    pub name: String,
    pub payload: T,
    pub context: JobContext,
}

#[derive(Clone, Debug)]
pub struct WorkerOutput<T> {
    pub payload: T,
    pub device_time_ms: f64,
}

pub trait ModelExecutor<I, O>: Send {
    fn run(&mut self, input: I, ctx: &JobContext) -> Result<WorkerOutput<O>>;
}
