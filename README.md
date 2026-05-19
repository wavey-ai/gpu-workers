# gpu-workers

Shared GPU worker runtime crates for Wavey services.

This workspace is the extraction point for runtime concerns that should not
live inside model-specific applications such as ASR or EnCodec.

## Crates

- `gpu-worker-core`
  - generic job metadata and executor traits
  - no ONNX, Torch, or upload-response assumptions
- `gpu-worker`
  - feature-gated facade crate for app services
  - re-exports the shared core, runtime, and upload-response adapters behind modules
- `gpu-worker-ort`
  - shared ONNX Runtime bootstrap
  - provider policy for CPU, CUDA, TensorRT, and CoreML
  - session construction and runtime discovery helpers
- `gpu-worker-torch`
  - shared libtorch/tch helpers
  - CUDA device, module loading, tensor construction, synchronization
- `gpu-worker-upload-response`
  - shared adapter over `upload-response`
  - local job abstraction for `request -> stage` and `stage -> response`
  - reusable local worker loop with claim/inflight/heartbeat handling
  - keeps transport concerns out of model workers

## Intended Layering

The intended dependency direction is:

1. transport/queue adapter
2. backend runtime
3. model-specific execution

Concretely:

- `upload-response` owns the generic stream/ring transport
- `gpu-worker::upload_response` adapts that transport into worker jobs
- `gpu-worker-ort` and `gpu-worker-torch` own backend runtime policy
- app crates such as `asr-onnx`, `asr-torch`, and `encodec-rs` should only
  own model semantics, preprocessing, and postprocessing

## Migration Status

Current first-phase extraction:

- `encodec-rs` uses `gpu-worker-ort` for ONNX session construction
- `gpu-worker::upload_response` provides the first reusable local worker job
  abstraction on top of named intermediate stages
- `asr-api` uses the facade crate for shared local and remote upload-response worker loops

Still to do:

- remote worker/job discovery in the upload-response adapter
- a generic worker loop/batching layer on top of `gpu-worker-core`
- thin app worker binaries that replace the remaining in-crate thread pools
