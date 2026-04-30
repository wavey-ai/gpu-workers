use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use http::Request;
use http_pack::stream::{StreamHeaders, StreamRequestHeaders, StreamResponseHeaders};
use tokio::task::JoinSet;
use tokio::time::{interval, Interval};
use tracing::{debug, error, info, warn};
use upload_response::{
    discover_ingress_origins, request_from_stream_headers, ActiveStreamInfo, RemoteIngressClient,
    RemoteRequestSlot, RemoteStageSlot, RemoteStreamInfo, RequestControl, StageTailSlot, TailSlot,
    UploadResponseService, WorkerHeartbeatUpdate,
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

    pub async fn source_last(&self) -> Result<usize> {
        match &self.spec.source {
            SourceLane::Request => self
                .service
                .request_last(self.stream_id)
                .ok_or_else(|| anyhow::anyhow!("request stream {} disappeared", self.stream_id)),
            SourceLane::Stage(stage) => self
                .service
                .stage_last(self.stream_id, stage)
                .await
                .ok_or_else(|| {
                    anyhow::anyhow!("stage {stage} stream {} disappeared", self.stream_id)
                }),
        }
    }

    pub fn source_reader_from(
        &self,
        last_slot: usize,
        poll_interval: Duration,
    ) -> LocalSourceReader {
        LocalSourceReader::new(self.clone(), last_slot, poll_interval)
    }

    pub async fn request(&self) -> Result<Option<Request<()>>> {
        match self.request_headers().await {
            Some(headers) => request_from_stream_headers(headers).map(Some),
            None => Ok(None),
        }
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
        match self.service.tail_request(self.stream_id, 1).await? {
            TailSlot::Headers(headers) => Some(headers),
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

#[derive(Clone)]
pub struct RemoteJob {
    client: RemoteIngressClient,
    worker_id: String,
    pub origin: String,
    pub stream_id: u64,
    pub spec: PipelineSpec,
}

impl RemoteJob {
    pub fn new(
        client: RemoteIngressClient,
        worker_id: impl Into<String>,
        origin: impl Into<String>,
        stream_id: u64,
        spec: PipelineSpec,
    ) -> Self {
        Self {
            client,
            worker_id: worker_id.into(),
            origin: origin.into(),
            stream_id,
            spec,
        }
    }

    pub fn client(&self) -> &RemoteIngressClient {
        &self.client
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn slot_bytes(&self) -> usize {
        self.client.slot_bytes()
    }

    pub fn inflight_key(&self) -> String {
        format!("{}#{}", self.origin, self.stream_id)
    }

    pub async fn source_last(&self) -> Result<usize> {
        match &self.spec.source {
            SourceLane::Request => Ok(self
                .client
                .request_last(&self.origin, self.stream_id)
                .await?),
            SourceLane::Stage(stage) => Ok(self
                .client
                .stage_last(&self.origin, self.stream_id, stage)
                .await?),
        }
    }

    pub fn source_reader_from(
        &self,
        last_slot: usize,
        poll_interval: Duration,
    ) -> RemoteSourceReader {
        RemoteSourceReader::new(self.clone(), last_slot, poll_interval)
    }

    pub async fn request(&self) -> Result<Option<Request<()>>> {
        self.client
            .request_headers(&self.origin, self.stream_id)
            .await
    }

    pub async fn tail(&self, slot_id: usize) -> Result<Option<SourceFrame>> {
        match &self.spec.source {
            SourceLane::Request => Ok(
                match self
                    .client
                    .request_slot(&self.origin, self.stream_id, slot_id)
                    .await?
                {
                    Some(RemoteRequestSlot::Headers(_)) => None,
                    Some(RemoteRequestSlot::Body(body)) => Some(SourceFrame::Body(body)),
                    Some(RemoteRequestSlot::Control(control)) => {
                        Some(SourceFrame::Control(control))
                    }
                    Some(RemoteRequestSlot::End) => Some(SourceFrame::End),
                    None => None,
                },
            ),
            SourceLane::Stage(stage) => Ok(
                match self
                    .client
                    .stage_slot(&self.origin, self.stream_id, stage, slot_id)
                    .await?
                {
                    Some(RemoteStageSlot::Head(head)) => Some(SourceFrame::StageHead(head)),
                    Some(RemoteStageSlot::Body(body)) => Some(SourceFrame::Body(body)),
                    Some(RemoteStageSlot::Control(control)) => Some(SourceFrame::Control(control)),
                    Some(RemoteStageSlot::End) => Some(SourceFrame::End),
                    None => None,
                },
            ),
        }
    }

    pub async fn stage_head(&self) -> Result<Option<Bytes>> {
        match self.tail(1).await? {
            Some(SourceFrame::StageHead(head)) => Ok(Some(head)),
            _ => Ok(None),
        }
    }

    pub async fn write_stage_head(&self, head: Bytes) -> Result<()> {
        match &self.spec.sink {
            SinkLane::Stage(stage) => {
                self.client
                    .write_stage_head(&self.origin, self.stream_id, stage, head)
                    .await
            }
            SinkLane::Response => anyhow::bail!("response sink does not accept opaque stage heads"),
        }
    }

    pub async fn write_response_headers(&self, headers: StreamResponseHeaders) -> Result<()> {
        match self.spec.sink {
            SinkLane::Response => {
                self.client
                    .write_response_headers(
                        &self.origin,
                        self.stream_id,
                        StreamHeaders::Response(headers),
                    )
                    .await
            }
            SinkLane::Stage(_) => anyhow::bail!("stage sink does not accept response headers"),
        }
    }

    pub async fn append_body(&self, body: Bytes) -> Result<()> {
        match &self.spec.sink {
            SinkLane::Stage(stage) => {
                self.client
                    .append_stage_body(&self.origin, self.stream_id, stage, body)
                    .await
            }
            SinkLane::Response => {
                self.client
                    .append_response_body(&self.origin, self.stream_id, body)
                    .await
            }
        }
    }

    pub async fn append_control(&self, control: RequestControl) -> Result<()> {
        match &self.spec.sink {
            SinkLane::Stage(stage) => {
                self.client
                    .append_stage_control(&self.origin, self.stream_id, stage, control)
                    .await
            }
            SinkLane::Response => anyhow::bail!("response sink does not support control markers"),
        }
    }

    pub async fn end(&self) -> Result<()> {
        match &self.spec.sink {
            SinkLane::Stage(stage) => {
                self.client
                    .end_stage(&self.origin, self.stream_id, stage)
                    .await
            }
            SinkLane::Response => self.client.end_response(&self.origin, self.stream_id).await,
        }
    }

    pub async fn release(&self) -> Result<()> {
        match &self.spec.sink {
            SinkLane::Stage(stage) => {
                self.client
                    .release_stage(&self.origin, self.stream_id, stage, &self.worker_id)
                    .await
            }
            SinkLane::Response => {
                self.client
                    .release_response(&self.origin, self.stream_id, &self.worker_id)
                    .await
            }
        }
    }
}

pub struct LocalSourceReader {
    job: LocalJob,
    poll: Interval,
    next_slot: usize,
    current_last: usize,
    finished: bool,
}

impl LocalSourceReader {
    fn new(job: LocalJob, last_slot: usize, poll_interval: Duration) -> Self {
        Self {
            job,
            poll: interval(poll_interval.max(Duration::from_millis(1))),
            next_slot: last_slot + 1,
            current_last: last_slot,
            finished: false,
        }
    }

    pub async fn next_frame(&mut self) -> Result<Option<SourceFrame>> {
        if self.finished {
            return Ok(None);
        }

        loop {
            if self.next_slot <= self.current_last {
                let slot_id = self.next_slot;
                self.next_slot += 1;
                match self.job.tail(slot_id).await {
                    Some(SourceFrame::End) => {
                        self.finished = true;
                        return Ok(Some(SourceFrame::End));
                    }
                    Some(frame) => return Ok(Some(frame)),
                    None => continue,
                }
            }

            self.poll.tick().await;
            self.current_last = self.job.source_last().await?;
        }
    }
}

pub struct RemoteSourceReader {
    job: RemoteJob,
    poll: Interval,
    next_slot: usize,
    current_last: usize,
    finished: bool,
}

impl RemoteSourceReader {
    fn new(job: RemoteJob, last_slot: usize, poll_interval: Duration) -> Self {
        Self {
            job,
            poll: interval(poll_interval.max(Duration::from_millis(1))),
            next_slot: last_slot + 1,
            current_last: last_slot,
            finished: false,
        }
    }

    pub async fn next_frame(&mut self) -> Result<Option<SourceFrame>> {
        if self.finished {
            return Ok(None);
        }

        loop {
            if self.next_slot <= self.current_last {
                let slot_id = self.next_slot;
                self.next_slot += 1;
                match self.job.tail(slot_id).await? {
                    Some(SourceFrame::End) => {
                        self.finished = true;
                        return Ok(Some(SourceFrame::End));
                    }
                    Some(frame) => return Ok(Some(frame)),
                    None => continue,
                }
            }

            self.poll.tick().await;
            self.current_last = self.job.source_last().await?;
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

#[derive(Debug, Clone)]
pub struct RemoteWorkerConfig {
    pub worker_id: String,
    pub heartbeat_stage: String,
    pub max_inflight: usize,
    pub poll_interval: Duration,
    pub discovery_interval: Duration,
    pub heartbeat_interval: Duration,
    pub ingress_urls: Vec<String>,
    pub discovery_dns: Option<String>,
    pub spec: PipelineSpec,
}

impl RemoteWorkerConfig {
    pub fn new(worker_id: impl Into<String>, spec: PipelineSpec) -> Self {
        let heartbeat_stage = spec.sink_stage_name();
        Self {
            worker_id: worker_id.into(),
            heartbeat_stage,
            max_inflight: 1,
            poll_interval: Duration::from_millis(100),
            discovery_interval: Duration::from_secs(5),
            heartbeat_interval: Duration::from_secs(1),
            ingress_urls: Vec::new(),
            discovery_dns: None,
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
pub trait RemoteJobProcessor: Send + Sync + 'static {
    async fn process(&self, job: RemoteJob) -> Result<()>;
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

pub async fn run_remote_worker_loop<P>(
    client: RemoteIngressClient,
    config: RemoteWorkerConfig,
    processor: Arc<P>,
) where
    P: RemoteJobProcessor,
{
    info!(
        worker_id = %config.worker_id,
        heartbeat_stage = %config.heartbeat_stage,
        max_inflight = config.max_inflight,
        poll_ms = config.poll_interval.as_millis(),
        discovery_ms = config.discovery_interval.as_millis(),
        heartbeat_ms = config.heartbeat_interval.as_millis(),
        source = ?config.spec.source,
        sink = ?config.spec.sink,
        "remote worker loop started"
    );

    let mut poll = interval(config.poll_interval.max(Duration::from_millis(1)));
    let mut discovery = interval(config.discovery_interval.max(Duration::from_millis(1)));
    let mut heartbeat = interval(config.heartbeat_interval.max(Duration::from_millis(1)));
    let mut inflight = HashSet::new();
    let mut tasks = JoinSet::new();
    let mut origins = Vec::new();
    let mut refresh_origins = true;
    let mut send_heartbeat = true;

    loop {
        tokio::select! {
            _ = poll.tick() => {}
            _ = discovery.tick() => {
                refresh_origins = true;
            }
            _ = heartbeat.tick() => {
                send_heartbeat = true;
            }
        }

        if refresh_origins {
            match discover_ingress_origins(&config.ingress_urls, config.discovery_dns.as_deref())
                .await
            {
                Ok(next) => {
                    if next != origins {
                        debug!(origins = ?next, "updated ingress origins");
                    }
                    origins = next;
                }
                Err(error) => {
                    warn!(error = %error, "failed to discover ingress origins");
                }
            }
            refresh_origins = false;
            send_heartbeat = true;
        }

        while let Some(joined) = tasks.try_join_next() {
            match joined {
                Ok(key) => {
                    inflight.remove(&key);
                    send_heartbeat = true;
                }
                Err(join_error) => {
                    error!(%join_error, "remote worker task failed");
                }
            }
        }

        if send_heartbeat {
            let heartbeat_update = config.heartbeat(inflight.len());
            for origin in &origins {
                if let Err(error) = client
                    .heartbeat_worker(origin, &config.worker_id, &heartbeat_update)
                    .await
                {
                    warn!(
                        origin,
                        worker_id = %config.worker_id,
                        error = %error,
                        "failed to publish remote worker capacity"
                    );
                }
            }
            send_heartbeat = false;
        }

        while inflight.len() < config.max_inflight {
            let Some(job) = claim_next_remote_job(&client, &config, &origins, &inflight).await
            else {
                break;
            };

            let _ = client
                .register_reader(&job.origin, job.stream_id, &config.worker_id)
                .await;

            let inflight_key = job.inflight_key();
            inflight.insert(inflight_key.clone());
            send_heartbeat = true;

            let processor = Arc::clone(&processor);
            let client = client.clone();
            let worker_id = config.worker_id.clone();
            tasks.spawn(async move {
                if let Err(error) = processor.process(job.clone()).await {
                    error!(
                        origin = %job.origin,
                        stream_id = job.stream_id,
                        error = %error,
                        "remote worker job failed"
                    );
                }
                let _ = job.release().await;
                let _ = client
                    .unregister_reader(&job.origin, job.stream_id, &worker_id)
                    .await;
                inflight_key
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

async fn claim_next_remote_job(
    client: &RemoteIngressClient,
    config: &RemoteWorkerConfig,
    origins: &[String],
    inflight: &HashSet<String>,
) -> Option<RemoteJob> {
    for origin in origins {
        let streams = match client.list_streams(origin).await {
            Ok(streams) => streams,
            Err(error) => {
                warn!(origin, error = %error, "failed to list remote streams");
                continue;
            }
        };

        for stream in streams {
            let inflight_key = format!("{}#{}", origin, stream.stream_id);
            if inflight.contains(&inflight_key) {
                continue;
            }
            if !remote_source_ready(&stream, &config.spec.source)
                || !remote_sink_available(&stream, &config.spec.sink)
            {
                continue;
            }

            let claimed = match &config.spec.sink {
                SinkLane::Stage(stage) => {
                    match client
                        .try_claim_stage(origin, stream.stream_id, stage, &config.worker_id)
                        .await
                    {
                        Ok(claimed) => claimed,
                        Err(error) => {
                            warn!(
                                origin,
                                stream_id = stream.stream_id,
                                stage,
                                error = %error,
                                "failed to claim remote stage"
                            );
                            continue;
                        }
                    }
                }
                SinkLane::Response => {
                    match client
                        .try_claim_response(origin, stream.stream_id, &config.worker_id)
                        .await
                    {
                        Ok(claimed) => claimed,
                        Err(error) => {
                            warn!(
                                origin,
                                stream_id = stream.stream_id,
                                error = %error,
                                "failed to claim remote response"
                            );
                            continue;
                        }
                    }
                }
            };

            if claimed {
                return Some(RemoteJob::new(
                    client.clone(),
                    config.worker_id.clone(),
                    origin.clone(),
                    stream.stream_id,
                    config.spec.clone(),
                ));
            }
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

fn remote_source_ready(stream: &RemoteStreamInfo, source: &SourceLane) -> bool {
    match source {
        SourceLane::Request => stream.request_last > 0,
        SourceLane::Stage(stage) => stream.stage_last(stage) > 0,
    }
}

fn remote_sink_available(stream: &RemoteStreamInfo, sink: &SinkLane) -> bool {
    match sink {
        SinkLane::Stage(stage) => {
            stream.stage_last(stage) == 0 && stream.stage_owner(stage).is_none()
        }
        SinkLane::Response => stream.response_owner.is_none(),
    }
}
