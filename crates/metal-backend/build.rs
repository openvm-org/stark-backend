// Metal shader build system
// Compiles .metal files → .air (intermediate) → .metallib (final library)
// The resulting .metallib is embedded into the binary via include_bytes!

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        // On non-macOS, create an empty metallib placeholder
        // so the crate can still compile (for documentation, CI on Linux, etc.)
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let metallib_path = out_dir.join("kernels.metallib");
        std::fs::write(&metallib_path, &[]).unwrap();
        println!(
            "cargo:rustc-env=METAL_KERNELS_PATH={}",
            metallib_path.display()
        );
        println!("cargo:warning=Metal shaders not compiled (non-macOS target)");
        return;
    }

    // Verify Metal compiler is available
    let metal_check = Command::new("xcrun")
        .args(["--sdk", "macosx", "metal", "--version"])
        .output();

    if metal_check.is_err() || !metal_check.unwrap().status.success() {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let metallib_path = out_dir.join("kernels.metallib");
        std::fs::write(&metallib_path, &[]).unwrap();
        println!(
            "cargo:rustc-env=METAL_KERNELS_PATH={}",
            metallib_path.display()
        );
        println!(
            "cargo:warning=Metal compiler not found. Install Xcode for Metal GPU support."
        );
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let metal_src_dir = manifest_dir.join("metal/src");
    let metal_include_dir = manifest_dir.join("metal/include");

    // Find all .metal files recursively
    let metal_files: Vec<PathBuf> = glob::glob(
        metal_src_dir
            .join("**/*.metal")
            .to_str()
            .unwrap(),
    )
    .unwrap()
    .filter_map(|e| e.ok())
    .collect();

    if metal_files.is_empty() {
        panic!("No .metal files found in {}", metal_src_dir.display());
    }

    // Set rerun-if-changed for all source and header files
    for f in &metal_files {
        println!("cargo:rerun-if-changed={}", f.display());
    }
    if let Ok(headers) = glob::glob(
        metal_include_dir
            .join("**/*.h")
            .to_str()
            .unwrap(),
    ) {
        for entry in headers.filter_map(|e| e.ok()) {
            println!("cargo:rerun-if-changed={}", entry.display());
        }
    }
    println!("cargo:rerun-if-changed=build.rs");

    // Stage 1: Compile each .metal -> .air
    let mut air_files = Vec::new();
    for src in &metal_files {
        // Use relative path with / replaced by _ to avoid collisions
        // e.g. logup_zerocheck/utils.metal -> logup_zerocheck_utils.air
        let relative = src.strip_prefix(&metal_src_dir).unwrap();
        let air_name = relative
            .to_str()
            .unwrap()
            .replace('/', "_")
            .replace(".metal", ".air");
        let air_path = out_dir.join(&air_name);

        let output = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-c",
                src.to_str().unwrap(),
                "-I",
                metal_include_dir.to_str().unwrap(),
                "-std=metal3.0",
                "-O2",
                "-o",
                air_path.to_str().unwrap(),
            ])
            .output()
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to run Metal compiler for {}: {}",
                    src.display(),
                    e
                )
            });

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "Metal compilation failed for {}:\n{}",
                src.display(),
                stderr
            );
        }

        air_files.push(air_path);
    }

    // Stage 2: Link all .air -> .metallib
    let metallib_path = out_dir.join("kernels.metallib");
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for air in &air_files {
        cmd.arg(air.to_str().unwrap());
    }
    cmd.args(["-o", metallib_path.to_str().unwrap()]);

    let output = cmd.output().expect("Failed to run metallib linker");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("metallib linking failed:\n{}", stderr);
    }

    println!(
        "cargo:rustc-env=METAL_KERNELS_PATH={}",
        metallib_path.display()
    );

    println!(
        "cargo:warning=Compiled {} Metal shaders into {}",
        metal_files.len(),
        metallib_path.display()
    );
}
