use std::path::Path;

use anyhow::Result;
use tch::{CModule, Cuda, Device, Kind, Tensor};

pub fn cuda_device(device_id: usize) -> Device {
    Device::Cuda(device_id)
}

pub fn load_module_on_cuda(path: impl AsRef<Path>, device_id: usize) -> Result<CModule> {
    Ok(CModule::load_on_device(path, cuda_device(device_id))?)
}

pub fn f32_tensor_on_cuda(data: &[f32], shape: &[i64], device_id: usize) -> Result<Tensor> {
    Ok(Tensor::f_from_slice(data)?
        .view(shape)
        .to_device(cuda_device(device_id)))
}

pub fn i64_tensor_on_cuda(data: &[i64], shape: &[i64], device_id: usize) -> Result<Tensor> {
    Ok(Tensor::f_from_slice(data)?
        .to_kind(Kind::Int64)
        .view(shape)
        .to_device(cuda_device(device_id)))
}

pub fn synchronize_cuda(device_id: usize) {
    Cuda::synchronize(device_id as i64);
}
