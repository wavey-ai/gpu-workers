use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use http_pack::stream::{StreamHeaders, StreamRequestHeaders, StreamResponseHeaders};
use tokio::task::JoinSet;
use tokio::time::interval;
use tracing::{error, info};
use upload_response::{
    ActiveStreamInfo, RequestControl, StageTailSlot, TailSlot, UploadResponseService,
    WorkerHeartbeatUpdate,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceLane {
    Request,
    Stage(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkLane {
    Stage(String),
    Response,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineSpec {
    pub source: SourceLane,
    pub sink: SinkLane,
}

impl PipelineSpec {
    pub fn sink_stage_name(&self) -> String {
        match &self.sink {
            SinkLane::Stage(stage) => stage.clone(),
            SinkLane::Response => "response".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceFrame {
    RequestHeaders(StreamRequestHeaders),
    StageHead(Bytes),
    Control(RequestControl),
    Body(Bytes),
    End,
}

#[derive(Clone)]
pub struct LocalJob {
    service: Arc<UploadResponseService>,
    worker_id: String,
    pub stream_id: u64,
    pub spec: PipelineSpec,
}

impl LocalJob {
    pub fn new(
        service: Arc<UploadResponseService>,
        worker_id: impl Into<String>,
        stream_id: u64,
        spec: PipelineSpec,
    ) -> Self {
        Self {
            service,
            worker_id: worker_id.into(),
            stream_id,
            spec,
        }
    }

    pub fn service(&self) -> &Arc<UploadResponseService> {
        &self.service
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn slot_bytes(&self) -> usize {
        self.service.config().slot_bytes()
    }

    pub async fn tail(&self, slot_id: usize) -> Option<SourceFrame> {
        match &self.spec.source {
            SourceLane::Request => {
                match self.service.tail_request(self.stream_id, slot_id).await? {
                    TailSlot::Headers(headers) => Some(SourceFrame::RequestHeaders(headers)),
                    TailSlot::Control(control) => Some(SourceFrame::Control(control)),
                    TailSlot::Body(body) => Some(SourceFrame::Body(body)),
                    TailSlot::End => Some(SourceFrame::End),
                }
            }
            SourceLane::Stage(stage) => {
                match self
                    .service
                    .tail_stage(self.stream_id, stage, slot_id)
                    .await?
                {
                    StageTailSlot::Head(head) => Some(SourceFrame::StageHead(head)),
                    StageTailSlot::Control(control) => Some(SourceFrame::Control(control)),
                    StageTailSlot::Body(body) => Some(SourceFrame::Body(body)),
                    StageTailSlot::End => Some(SourceFrame::End),
                }
            }
        }
    }

    pub async fn request_headers(&self) -> Option<StreamRequestHeaders> {
        match self.tail(1).await? {
            SourceFrame::RequestHeaders(headers) => Some(headers),
            _ => None,
        }
    }

    pub async fn stage_head(&self) -> Option<Bytes> {
        match self.tail(1).await? {
            SourceFrame::StageHead(head) => Some(head),
            _ => None,
        }
    }

    pub async fn write_stage_head(&self, head: Bytes) -> Result<(), String> {
        match &self.spec.sink {
            SinkLane::Stage(stage) => {
                self.service
                    .write_stage_head(self.stream_id, stage, head)
                    .await
            }
            SinkLane::Response => {
                Err("response sink does not accept opaque stage heads".to_string())
            }
        }
    }

    pub async fn write_response_headers(
        &self,
        headers: StreamResponseHeaders,
    ) -> Result<(), String> {
        match self.spec.sink {
            SinkLane::Response => {
                self.service
                    .write_response_headers(self.stream_id, StreamHeaders::Response(headers))
                    .await
            }
            SinkLane::Stage(_) => Err("stage sink does not accept response headers".to_string()),
        }
    }

    pub async fn append_body(&self, body: Bytes) -> Result<(), String> {
        match &self.spec.sink {
            SinkLane::Stage(stage) => {
                self.service
                    .append_stage_body(self.stream_id, stage, body)
                    .await
            }
            SinkLane::Response => {
                self.service
                    .append_response_body(self.stream_id, body)
                    .await
            }
        }
    }

    pub async fn append_control(&self, control: RequestControl) -> Result<(), String> {
        match &self.spec.sink {
            SinkLane::Stage(stage) => {
                self.service
                    .append_stage_control(self.stream_id, stage, control)
                    .await
            }
            SinkLane::Response => Err("response sink does not support control markers".to_string()),
        }
    }

    pub async fn end(&self) -> Result<(), String> {
        match &self.spec.sink {
            SinkLane::Stage(stage) => self.service.end_stage(self.stream_id, stage).await,
            SinkLane::Response => self.service.end_response(self.stream_id).await,
        }
    }

    pub async fn release(&self) -> bool {
        match &self.spec.sink {
            SinkLane::Stage(stage) => {
                self.service
                    .release_stage(self.stream_id, stage, &self.worker_id)
                    .await
            }
            SinkLane::Response => {
                self.service
                    .release_response(self.stream_id, &self.worker_id)
                    .await
            }
        }
    }
}

pub async fn claim_local_job(
    service: Arc<UploadResponseService>,
    spec: &PipelineSpec,
    worker_id: &str,
) -> Option<LocalJob> {
    for stream in service.active_streams().await {
        if !source_ready(&stream, &spec.source) || !sink_available(&stream, &spec.sink) {
            continue;
        }

        let claimed = match &spec.sink {
            SinkLane::Stage(stage) => {
                service
                    .try_claim_stage(stream.stream_id, stage, worker_id)
                    .await
            }
            SinkLane::Response => {
                service
                    .try_claim_response(stream.stream_id, worker_id)
                    .await
            }
        };
        if !claimed {
            continue;
        }

        return Some(LocalJob::new(
            Arc::clone(&service),
            worker_id,
            stream.stream_id,
            spec.clone(),
        ));
    }

    None
}

#[derive(Debug, Clone)]
pub struct LocalWorkerConfig {
    pub worker_id: String,
    pub heartbeat_stage: String,
    pub max_inflight: usize,
    pub poll_interval: Duration,
    pub heartbeat_interval: Duration,
    pub spec: PipelineSpec,
}

impl LocalWorkerConfig {
    pub fn new(worker_id: impl Into<String>, spec: PipelineSpec) -> Self {
        let heartbeat_stage = spec.sink_stage_name();
        Self {
            worker_id: worker_id.into(),
            heartbeat_stage,
            max_inflight: 1,
            poll_interval: Duration::from_millis(100),
            heartbeat_interval: Duration::from_secs(1),
            spec,
        }
    }

    fn heartbeat(&self, inflight: usize) -> WorkerHeartbeatUpdate {
        let inflight = inflight.min(self.max_inflight);
        WorkerHeartbeatUpdate {
            stage: self.heartbeat_stage.clone(),
            max_inflight: self.max_inflight,
            inflight,
            available_slots: self.max_inflight.saturating_sub(inflight),
        }
    }
}

#[async_trait]
pub trait LocalJobProcessor: Send + Sync + 'static {
    async fn process(&self, job: LocalJob) -> Result<()>;
}

pub async fn run_local_worker_loop<P>(
    service: Arc<UploadResponseService>,
    config: LocalWorkerConfig,
    processor: Arc<P>,
) where
    P: LocalJobProcessor,
{
    info!(
        worker_id = %config.worker_id,
        heartbeat_stage = %config.heartbeat_stage,
        max_inflight = config.max_inflight,
        poll_ms = config.poll_interval.as_millis(),
        heartbeat_ms = config.heartbeat_interval.as_millis(),
        source = ?config.spec.source,
        sink = ?config.spec.sink,
        "local worker loop started"
    );

    let mut poll = interval(config.poll_interval.max(Duration::from_millis(1)));
    let mut heartbeat = interval(config.heartbeat_interval.max(Duration::from_millis(1)));
    let mut inflight = HashSet::new();
    let mut tasks = JoinSet::new();
    let mut send_heartbeat = true;

    loop {
        tokio::select! {
            _ = poll.tick() => {}
            _ = heartbeat.tick() => {
                send_heartbeat = true;
            }
        }

        while let Some(joined) = tasks.try_join_next() {
            match joined {
                Ok(stream_id) => {
                    inflight.remove(&stream_id);
                    send_heartbeat = true;
                }
                Err(join_error) => {
                    error!(%join_error, "local worker task failed");
                }
            }
        }

        if send_heartbeat {
            service
                .upsert_worker_heartbeat(&config.worker_id, config.heartbeat(inflight.len()))
                .await;
            send_heartbeat = false;
        }

        while inflight.len() < config.max_inflight {
            let Some(job) = claim_next_local_job(Arc::clone(&service), &config, &inflight).await
            else {
                break;
            };

            let _ = service
                .register_reader(job.stream_id, &config.worker_id)
                .await;

            inflight.insert(job.stream_id);
            send_heartbeat = true;

            let processor = Arc::clone(&processor);
            let service = Arc::clone(&service);
            let worker_id = config.worker_id.clone();
            tasks.spawn(async move {
                let stream_id = job.stream_id;
                if let Err(error) = processor.process(job.clone()).await {
                    error!(stream_id, error = %error, "local worker job failed");
                }
                let _ = job.release().await;
                let _ = service.unregister_reader(stream_id, &worker_id).await;
                stream_id
            });
        }
    }
}

async fn claim_next_local_job(
    service: Arc<UploadResponseService>,
    config: &LocalWorkerConfig,
    inflight: &HashSet<u64>,
) -> Option<LocalJob> {
    for stream in service.active_streams().await {
        if inflight.contains(&stream.stream_id) {
            continue;
        }
        if !source_ready(&stream, &config.spec.source)
            || !sink_available(&stream, &config.spec.sink)
        {
            continue;
        }

        let claimed = match &config.spec.sink {
            SinkLane::Stage(stage) => {
                service
                    .try_claim_stage(stream.stream_id, stage, &config.worker_id)
                    .await
            }
            SinkLane::Response => {
                service
                    .try_claim_response(stream.stream_id, &config.worker_id)
                    .await
            }
        };

        if claimed {
            return Some(LocalJob::new(
                Arc::clone(&service),
                config.worker_id.clone(),
                stream.stream_id,
                config.spec.clone(),
            ));
        }
    }

    None
}

fn source_ready(stream: &ActiveStreamInfo, source: &SourceLane) -> bool {
    match source {
        SourceLane::Request => stream.request_last > 0,
        SourceLane::Stage(stage) => stream.stage_last(stage) > 0,
    }
}

fn sink_available(stream: &ActiveStreamInfo, sink: &SinkLane) -> bool {
    match sink {
        SinkLane::Stage(stage) => {
            stream.stage_last(stage) == 0 && stream.stage_owner(stage).is_none()
        }
        SinkLane::Response => stream.response_last == 0 && stream.response_owner.is_none(),
    }
}
