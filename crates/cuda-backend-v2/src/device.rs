use getset::CopyGetters;
use openvm_cuda_common::common::get_device;
use stark_backend_v2::keygen::types::SystemParams;

#[derive(Clone, Copy, CopyGetters)]
pub struct GpuDeviceV2 {
    #[getset(get_copy = "pub")]
    pub config: SystemParams,
    pub id: u32,
}

impl GpuDeviceV2 {
    pub fn new(config: SystemParams) -> Self {
        Self {
            config,
            id: get_device().unwrap() as u32,
        }
    }
}
