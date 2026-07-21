//! Prints the generated CUDA for `ntt_shared_module(log_n)`.

fn main() {
    let log_n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let module = crypto_compiler::kernels::ntt_shared_module(log_n);
    let program = crypto_compiler::canonicalize::canonicalize(module).unwrap();
    let mut kprog = crypto_compiler::lower::lower(&program).unwrap();
    crypto_compiler::passes::layout_infer(&mut kprog);
    crypto_compiler::passes::insert_sync(&mut kprog);
    println!("{}", crypto_compiler::codegen::generate_cuda(&kprog));
}
