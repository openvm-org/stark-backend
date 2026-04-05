use openvm_cuda_backend::{ntt::batch_ntt, prelude::F};
use openvm_cuda_common::{
    common::get_device,
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::{CudaStream, DeviceContext, StreamGuard},
};
use p3_field::PrimeCharacteristicRing;

#[test]
#[ignore] // Run explicitly: requires large GPU memory and significant runtime.
fn ntt_roundtrip_max_log_domain_size() {
    const LOG_N: u32 = 27;
    let ctx = DeviceContext {
        device_id: get_device().unwrap() as u32,
        stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
    };
    let n = 1usize << LOG_N;
    let mut host = Vec::<F>::with_capacity(n);

    for i in 0..n {
        host.push(F::from_usize(i));
    }

    let mut device = DeviceBuffer::<F>::with_capacity_on(n, &ctx);
    host.copy_to_on(&mut device, &ctx)
        .expect("host->device copy failed");

    batch_ntt(&device, LOG_N, 0, 1, true, false, &ctx);
    batch_ntt(&device, LOG_N, 0, 1, true, true, &ctx);

    let output = device.to_host_on(&ctx).expect("device->host copy failed");
    for (i, got) in output.iter().enumerate() {
        assert_eq!(*got, host[i], "mismatch at index {}", i);
    }
}
