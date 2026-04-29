use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};

pub use ort::execution_providers::{
    CPUExecutionProvider, CUDAExecutionProvider, CoreML, ExecutionProvider,
    ExecutionProviderDispatch, TensorRT, coreml,
};
pub use ort::inputs;
pub use ort::logging::LogLevel;
pub use ort::session::Session;
pub use ort::session::builder::GraphOptimizationLevel;
pub use ort::value::Tensor as OrtTensor;
pub use ort::value::ValueType;

pub fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow::anyhow!(error.to_string())
}

#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub optimization_level: GraphOptimizationLevel,
    pub log_level: LogLevel,
    pub intra_threads: usize,
}

impl SessionConfig {
    pub fn new(
        optimization_level: GraphOptimizationLevel,
        log_level: LogLevel,
        intra_threads: usize,
    ) -> Self {
        Self {
            optimization_level,
            log_level,
            intra_threads,
        }
    }
}

pub fn default_intra_threads(limit: usize) -> usize {
    std::thread::available_parallelism()
        .map(|value| value.get().min(limit.max(1)))
        .unwrap_or(1)
}

pub fn build_session(
    path: &Path,
    providers: impl AsRef<[ExecutionProviderDispatch]>,
    config: &SessionConfig,
) -> Result<Session> {
    ort::session::Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(config.optimization_level)
        .map_err(ort_error)?
        .with_log_level(config.log_level)
        .map_err(ort_error)?
        .with_execution_providers(providers)
        .map_err(ort_error)?
        .with_intra_threads(config.intra_threads)
        .map_err(ort_error)?
        .commit_from_file(path)
        .map_err(ort_error)
}

pub fn default_dynamic_runtime_candidates(env_keys: &[&str]) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    for key in env_keys {
        if let Ok(value) = env::var(key) {
            let path = PathBuf::from(value);
            if !path.as_os_str().is_empty() {
                candidates.push(path);
            }
        }
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("libonnxruntime.so"));
            candidates.push(dir.join("deps").join("libonnxruntime.so"));
            candidates.push(dir.join("lib").join("libonnxruntime.so"));
        }
    }

    candidates.push(PathBuf::from("/usr/local/lib/libonnxruntime.so"));
    candidates.push(PathBuf::from("/usr/lib/x86_64-linux-gnu/libonnxruntime.so"));

    let mut unique = Vec::new();
    let mut seen = HashSet::new();
    for path in candidates {
        if seen.insert(path.clone()) {
            unique.push(path);
        }
    }
    unique
}

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

pub fn ensure_dynamic_runtime_from_env(env_keys: &[&str]) -> Result<()> {
    let candidates = default_dynamic_runtime_candidates(env_keys);
    let result = ORT_INIT.get_or_init(|| {
        let mut errors = Vec::new();
        for candidate in &candidates {
            if !candidate.exists() {
                continue;
            }

            match ort::init_from(candidate) {
                Ok(builder) => {
                    let _created = builder.commit();
                    return Ok(());
                }
                Err(error) => errors.push(format!("{}: {}", candidate.display(), error)),
            }
        }

        if errors.is_empty() {
            Err("failed to locate libonnxruntime.so".to_string())
        } else {
            Err(format!(
                "failed to initialize libonnxruntime.so: {}",
                errors.join(" | ")
            ))
        }
    });

    result
        .as_ref()
        .map_err(|error| anyhow::anyhow!(error.clone()))?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionTarget {
    Cpu,
    Cuda {
        device_id: i32,
    },
    CoreMl {
        compute_units: CoreMlComputeUnits,
        model_cache_dir: Option<PathBuf>,
        low_precision_accumulation_on_gpu: bool,
    },
    TensorRt {
        device_id: i32,
        fp16: bool,
        engine_cache_path: Option<PathBuf>,
        timing_cache_path: Option<PathBuf>,
    },
}

impl Default for ExecutionTarget {
    fn default() -> Self {
        Self::Cpu
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoreMlComputeUnits {
    All,
    CpuAndNeuralEngine,
    CpuAndGpu,
    CpuOnly,
}

impl From<CoreMlComputeUnits> for coreml::ComputeUnits {
    fn from(value: CoreMlComputeUnits) -> Self {
        match value {
            CoreMlComputeUnits::All => Self::All,
            CoreMlComputeUnits::CpuAndNeuralEngine => Self::CPUAndNeuralEngine,
            CoreMlComputeUnits::CpuAndGpu => Self::CPUAndGPU,
            CoreMlComputeUnits::CpuOnly => Self::CPUOnly,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreMlProviderOptions {
    pub compute_units: CoreMlComputeUnits,
    pub model_cache_dir: Option<PathBuf>,
    pub low_precision_accumulation_on_gpu: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRtProviderOptions {
    pub device_id: i32,
    pub fp16: bool,
    pub engine_cache_path: Option<PathBuf>,
    pub engine_cache_prefix: Option<String>,
    pub timing_cache_path: Option<PathBuf>,
    pub profile_min_shapes: Option<String>,
    pub profile_opt_shapes: Option<String>,
    pub profile_max_shapes: Option<String>,
    pub max_workspace_size: Option<usize>,
    pub builder_optimization_level: Option<u8>,
    pub force_sequential_engine_build: bool,
    pub layer_norm_fp32_fallback: bool,
    pub detailed_build_log: bool,
}

impl Default for TensorRtProviderOptions {
    fn default() -> Self {
        Self {
            device_id: 0,
            fp16: false,
            engine_cache_path: None,
            engine_cache_prefix: None,
            timing_cache_path: None,
            profile_min_shapes: None,
            profile_opt_shapes: None,
            profile_max_shapes: None,
            max_workspace_size: None,
            builder_optimization_level: None,
            force_sequential_engine_build: true,
            layer_norm_fp32_fallback: false,
            detailed_build_log: false,
        }
    }
}

pub fn cpu_provider() -> ExecutionProviderDispatch {
    CPUExecutionProvider::default().build()
}

pub fn cuda_provider(device_id: i32, error_on_failure: bool) -> ExecutionProviderDispatch {
    let provider = CUDAExecutionProvider::default()
        .with_device_id(device_id)
        .build();
    if error_on_failure {
        provider.error_on_failure()
    } else {
        provider.fail_silently()
    }
}

pub fn coreml_provider(
    options: &CoreMlProviderOptions,
    error_on_failure: bool,
) -> Result<ExecutionProviderDispatch> {
    let base = CoreML::default();
    if !base.is_available().unwrap_or(false) {
        bail!("CoreML Execution Provider is not available");
    }

    let mut provider = base
        .with_compute_units(options.compute_units.into())
        .with_specialization_strategy(coreml::SpecializationStrategy::FastPrediction);
    if let Some(path) = &options.model_cache_dir {
        fs::create_dir_all(path)
            .with_context(|| format!("failed to create CoreML cache dir {}", path.display()))?;
        provider = provider.with_model_cache_dir(path.display().to_string());
    }
    if options.low_precision_accumulation_on_gpu {
        provider = provider.with_low_precision_accumulation_on_gpu(true);
    }

    let built = provider.build();
    Ok(if error_on_failure {
        built.error_on_failure()
    } else {
        built.fail_silently()
    })
}

pub fn tensorrt_provider(
    options: &TensorRtProviderOptions,
    fail_silently: bool,
) -> Result<ExecutionProviderDispatch> {
    let mut provider = TensorRT::default()
        .with_device_id(options.device_id)
        .with_engine_cache(true)
        .with_force_sequential_engine_build(options.force_sequential_engine_build)
        .with_timing_cache(true);
    if options.fp16 {
        provider = provider.with_fp16(true);
    }
    if let Some(prefix) = &options.engine_cache_prefix {
        provider = provider.with_engine_cache_prefix(prefix);
    }
    if let Some(level) = options.builder_optimization_level {
        provider = provider.with_builder_optimization_level(level);
    }
    if let Some(workspace) = options.max_workspace_size {
        provider = provider.with_max_workspace_size(workspace);
    }
    if options.layer_norm_fp32_fallback {
        provider = provider.with_layer_norm_fp32_fallback(true);
    }
    if options.detailed_build_log {
        provider = provider.with_detailed_build_log(true);
    }
    if let Some(path) = &options.engine_cache_path {
        fs::create_dir_all(path).with_context(|| {
            format!(
                "failed to create TensorRT engine cache dir {}",
                path.display()
            )
        })?;
        provider = provider.with_engine_cache_path(path.display().to_string());
    }
    if let Some(path) = &options.timing_cache_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create TensorRT timing cache dir {}",
                    parent.display()
                )
            })?;
        }
        provider = provider.with_timing_cache_path(path.display().to_string());
    }
    if let Some(shapes) = &options.profile_min_shapes {
        provider = provider.with_profile_min_shapes(shapes);
    }
    if let Some(shapes) = &options.profile_opt_shapes {
        provider = provider.with_profile_opt_shapes(shapes);
    }
    if let Some(shapes) = &options.profile_max_shapes {
        provider = provider.with_profile_max_shapes(shapes);
    }

    let built = provider.build();
    Ok(if fail_silently {
        built.fail_silently()
    } else {
        built.error_on_failure()
    })
}

pub fn providers_for_target(
    target: &ExecutionTarget,
    include_cpu_fallback: bool,
) -> Result<Vec<ExecutionProviderDispatch>> {
    let mut providers = Vec::new();
    match target {
        ExecutionTarget::Cpu => {
            providers.push(cpu_provider());
        }
        ExecutionTarget::Cuda { device_id } => {
            providers.push(cuda_provider(*device_id, true));
            if include_cpu_fallback {
                providers.push(cpu_provider());
            }
        }
        ExecutionTarget::CoreMl {
            compute_units,
            model_cache_dir,
            low_precision_accumulation_on_gpu,
        } => {
            providers.push(coreml_provider(
                &CoreMlProviderOptions {
                    compute_units: *compute_units,
                    model_cache_dir: model_cache_dir.clone(),
                    low_precision_accumulation_on_gpu: *low_precision_accumulation_on_gpu,
                },
                true,
            )?);
            if include_cpu_fallback {
                providers.push(cpu_provider());
            }
        }
        ExecutionTarget::TensorRt {
            device_id,
            fp16,
            engine_cache_path,
            timing_cache_path,
        } => {
            providers.push(tensorrt_provider(
                &TensorRtProviderOptions {
                    device_id: *device_id,
                    fp16: *fp16,
                    engine_cache_path: engine_cache_path.clone(),
                    timing_cache_path: timing_cache_path.clone(),
                    ..TensorRtProviderOptions::default()
                },
                false,
            )?);
            providers.push(cuda_provider(*device_id, true));
            if include_cpu_fallback {
                providers.push(cpu_provider());
            }
        }
    }
    Ok(providers)
}

pub fn build_session_from_target(
    path: &Path,
    target: &ExecutionTarget,
    config: &SessionConfig,
    include_cpu_fallback: bool,
) -> Result<Session> {
    let providers = providers_for_target(target, include_cpu_fallback)?;
    build_session(path, providers, config)
}

pub fn concrete_shape(value_type: &ValueType) -> Option<Vec<usize>> {
    let shape = value_type.tensor_shape()?;
    Some(
        shape
            .iter()
            .map(|dim| if *dim > 0 { *dim as usize } else { 1 })
            .collect(),
    )
}

pub fn last_positive_dim(value_type: &ValueType) -> Option<usize> {
    value_type
        .tensor_shape()?
        .iter()
        .rev()
        .find(|dim| **dim > 0)
        .map(|dim| *dim as usize)
}
