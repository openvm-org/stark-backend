//! Prints the generated CUDA for `ntt_shared_module(log_n)`.

use crypto_compiler::passes;

fn main() {
    let log_n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let module = crypto_compiler::kernels::ntt_shared_module(log_n);
    let types = passes::type_infer(&module).unwrap();
    let program = passes::canonicalize(module, types).unwrap();
    let scratch = passes::plan_global_scratch(&program).unwrap();
    let mut kprog = passes::lower_to_kir(&program, &scratch).unwrap();
    passes::layout_infer(&mut kprog);
    passes::insert_sync(&mut kprog);
    println!("{}", passes::codegen(&kprog).unwrap());
}
