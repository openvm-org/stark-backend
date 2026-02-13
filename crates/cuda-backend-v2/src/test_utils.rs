pub fn assert_eq_device_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    a: &DeviceMatrix<T>,
    b: &DeviceMatrix<T>,
) {
    assert_eq!(a.height(), b.height());
    assert_eq!(a.width(), b.width());
    assert_eq!(a.buffer().len(), b.buffer().len());
    let a_host = a.to_host().unwrap();
    let b_host = b.to_host().unwrap();
    for r in 0..a.height() {
        for c in 0..a.width() {
            assert_eq!(
                a_host[c * a.height() + r],
                b_host[c * b.height() + r],
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}

pub fn assert_eq_host_and_device_matrix<T: Clone + Send + Sync + PartialEq + Debug>(
    cpu: Arc<RowMajorMatrix<T>>,
    gpu: &DeviceMatrix<T>,
) {
    assert_eq!(gpu.width(), cpu.width());
    assert_eq!(gpu.height(), cpu.height());
    let gpu = gpu.to_host().unwrap();
    for r in 0..cpu.height() {
        for c in 0..cpu.width() {
            assert_eq!(
                gpu[c * cpu.height() + r],
                cpu.get(r, c).expect("matrix index out of bounds"),
                "Mismatch at row {} column {}",
                r,
                c
            );
        }
    }
}
