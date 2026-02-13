use openvm_cuda_backend::{ntt::batch_ntt, prelude::F};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
};
use p3_field::PrimeCharacteristicRing;

#[test]
#[ignore] // Run explicitly: requires large GPU memory and significant runtime.
fn ntt_roundtrip_max_log_domain_size() {
    const LOG_N: u32 = 27;
    let n = 1usize << LOG_N;
    let mut host = Vec::<F>::with_capacity(n);

    for i in 0..n {
        host.push(F::from_usize(i));
    }

    let mut device = DeviceBuffer::<F>::with_capacity(n);
    host.copy_to(&mut device).expect("host->device copy failed");

    batch_ntt(&device, LOG_N, 0, 1, true, false);
    batch_ntt(&device, LOG_N, 0, 1, true, true);

    let output = device.to_host().expect("device->host copy failed");
    for (i, got) in output.iter().enumerate() {
        assert_eq!(*got, host[i], "mismatch at index {}", i);
    }
}
