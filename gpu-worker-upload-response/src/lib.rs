use std::sync::Arc;

use bytes::Bytes;
use http_pack::stream::{StreamHeaders, StreamRequestHeaders, StreamResponseHeaders};
use upload_response::{
    ActiveStreamInfo, RequestControl, StageTailSlot, TailSlot, UploadResponseService,
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
