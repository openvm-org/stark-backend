use std::{env, path::Path, process::Command};

/// CUDA builder configuration
#[derive(Debug, Clone)]
pub struct CudaBuilder {
    include_paths: Vec<String>,
    source_files: Vec<String>,
    watch_paths: Vec<String>,
    watch_globs: Vec<String>,
    library_name: String,
    cuda_arch: Vec<String>,
    cuda_opt_level: Option<String>,
    custom_flags: Vec<String>,
    link_libraries: Vec<String>,
    link_search_paths: Vec<String>,
}

impl Default for CudaBuilder {
    fn default() -> Self {
        let mut link_search_paths = Vec::new();
        if let Ok(cuda_lib_dir) = env::var("CUDA_LIB_DIR") {
            link_search_paths.push(cuda_lib_dir);
        } else {
            link_search_paths.push("/usr/local/cuda/lib64".to_string());
        }

        Self {
            include_paths: Vec::new(),
            source_files: Vec::new(),
            watch_paths: vec!["build.rs".to_string()],
            watch_globs: Vec::new(),
            library_name: String::new(),
            cuda_arch: Vec::new(),
            cuda_opt_level: None,
            custom_flags: vec![
                "--std=c++17".to_string(),
                "--expt-relaxed-constexpr".to_string(),
                "-Xfatbin=-compress-all".to_string(),
                "--default-stream=per-thread".to_string(),
            ],
            link_libraries: vec!["cudart".to_string(), "cuda".to_string()],
            link_search_paths,
        }
    }
}

impl CudaBuilder {
    /// Create a new CudaBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the library name (useful when cloning from a template)
    pub fn library_name(mut self, name: &str) -> Self {
        self.library_name = name.to_string();
        self
    }

    /// Add include path
    pub fn include<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.include_paths.push(path_str.clone());
        self.watch_paths.push(path_str);
        self
    }

    /// Add include path from another crate's exported include
    pub fn include_from_dep(mut self, dep_env_var: &str) -> Self {
        if let Ok(path) = env::var(dep_env_var) {
            self.include_paths.push(path);
        }
        self
    }

    /// Add source file
    pub fn file<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.source_files.push(path_str.clone());
        self.watch_paths.push(path_str);
        self
    }

    /// Add multiple source files
    pub fn files<P: AsRef<Path>, I: IntoIterator<Item = P>>(mut self, paths: I) -> Self {
        for path in paths {
            let path_str = path.as_ref().to_string_lossy().to_string();
            self.source_files.push(path_str.clone());
            self.watch_paths.push(path_str);
        }
        self
    }

    /// Add multiple source files matching a glob pattern
    pub fn files_from_glob(mut self, pattern: &str) -> Self {
        self.watch_globs.push(pattern.to_string());
        for path in glob::glob(pattern).expect("Invalid glob pattern").flatten() {
            if path.is_file() && path.extension().is_some_and(|ext| ext == "cu") {
                self.source_files.push(path.to_string_lossy().to_string());
            }
        }
        self
    }

    /// Watch a specific path for changes
    pub fn watch<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.watch_paths
            .push(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Watch paths matching a glob pattern
    pub fn watch_glob(mut self, pattern: &str) -> Self {
        self.watch_globs.push(pattern.to_string());
        self
    }

    /// Set CUDA architecture (e.g., "75", "80")
    pub fn cuda_arch(mut self, arch: &str) -> Self {
        self.cuda_arch = vec![arch.to_string()];
        self
    }

    /// Set multiple CUDA architectures  
    pub fn cuda_archs(mut self, archs: Vec<&str>) -> Self {
        self.cuda_arch = archs.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Set CUDA optimization level (0-3)
    pub fn cuda_opt_level(mut self, level: u8) -> Self {
        self.cuda_opt_level = Some(level.to_string());
        self
    }

    /// Add custom compiler flag
    pub fn flag(mut self, flag: &str) -> Self {
        self.custom_flags.push(flag.to_string());
        self
    }

    /// Add library to link
    pub fn link_lib(mut self, lib: &str) -> Self {
        self.link_libraries.push(lib.to_string());
        self
    }

    /// Add library search path
    pub fn link_search<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.link_search_paths
            .push(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Build the CUDA library
    pub fn build(self) {
        // Validation
        self.validate();

        // Set up rerun conditions
        self.setup_rerun_conditions();

        // Get or detect CUDA architecture
        let cuda_archs = self.get_cuda_arch();

        // Create cc::Build
        let mut builder = cc::Build::new();
        builder.cuda(true);

        // Handle CUDA_DEBUG=1
        self.handle_debug_shortcuts(&mut builder);

        // Get optimization level
        let cuda_opt_level = self.get_cuda_opt_level();

        // Add include paths
        for include in &self.include_paths {
            builder.include(include);
        }

        // Add CUDA_PATH include if available
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            builder.include(format!("{}/include", cuda_path));
        }

        // Add custom flags
        for flag in &self.custom_flags {
            builder.flag(flag);
        }

        // Add SASS code for each architecture
        for arch in &cuda_archs {
            builder
                .flag("-gencode")
                .flag(format!("arch=compute_{},code=sm_{}", arch, arch));
        }

        // Add PTX for the highest architecture (forward compatibility)
        // This allows the code to run on future GPUs
        if let Some(max_arch) = cuda_archs.iter().max() {
            builder.flag("-gencode").flag(format!(
                "arch=compute_{},code=compute_{}",
                max_arch, max_arch
            ));
        }

        // Add parallel jobs flag
        builder.flag(nvcc_parallel_jobs());

        // Set optimization and debug flags
        if cuda_opt_level == "0" {
            builder.debug(true).flag("-O0");
        } else {
            builder
                .debug(false)
                .flag(format!("--ptxas-options=-O{}", cuda_opt_level));
        }

        // Add source files
        for file in &self.source_files {
            builder.file(file);
        }

        // Compile
        builder.compile(&self.library_name);
    }

    /// Validate the builder configuration
    fn validate(&self) {
        if self.library_name.is_empty() {
            panic!(
                "Library name must be set using .library_name(\"name\") before calling .build()"
            );
        }

        if self.source_files.is_empty() {
            panic!("At least one source file must be added using .file() or .files() before calling .build()");
        }

        // Validate that source files exist (optional, but helpful)
        for file in &self.source_files {
            if !Path::new(file).exists() {
                eprintln!("cargo:warning=CUDA source file does not exist: {}", file);
            }
        }

        // Validate include paths exist (optional warning)
        for include in &self.include_paths {
            if !Path::new(include).exists() {
                eprintln!("cargo:warning=Include path does not exist: {}", include);
            }
        }
    }

    pub fn emit_link_directives(&self) {
        for path in &self.link_search_paths {
            println!("cargo:rustc-link-search=native={}", path);
        }
        for lib in &self.link_libraries {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }

    fn setup_rerun_conditions(&self) {
        // Standard rerun conditions
        println!("cargo:rerun-if-env-changed=CUDA_ARCH");
        println!("cargo:rerun-if-env-changed=CUDA_OPT_LEVEL");
        println!("cargo:rerun-if-env-changed=CUDA_DEBUG");
        println!("cargo:rerun-if-env-changed=NVCC_THREADS");

        // Watch specific paths
        for path in &self.watch_paths {
            println!("cargo:rerun-if-changed={}", path);
        }

        // Watch glob patterns
        for pattern in &self.watch_globs {
            watch_glob(pattern);
        }
    }

    fn get_cuda_arch(&self) -> Vec<String> {
        if !self.cuda_arch.is_empty() {
            return self.cuda_arch.clone();
        }

        // Check environment variable
        if let Ok(env_archs) = env::var("CUDA_ARCH") {
            return env_archs
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }

        // Auto-detect current GPU
        vec![detect_cuda_arch()]
    }

    fn get_cuda_opt_level(&self) -> String {
        if let Some(level) = &self.cuda_opt_level {
            return level.clone();
        }

        env::var("CUDA_OPT_LEVEL").unwrap_or_else(|_| "3".to_string())
    }

    fn handle_debug_shortcuts(&self, builder: &mut cc::Build) {
        if env::var("CUDA_DEBUG").map(|v| v == "1").unwrap_or(false) {
            env::set_var("CUDA_OPT_LEVEL", "0");
            env::set_var("CUDA_LAUNCH_BLOCKING", "1");
            env::set_var("RUST_BACKTRACE", "full");
            env::set_var("CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "1");
            env::set_var("CUDA_DEVICE_WAITS_ON_EXCEPTION", "1");
            
            println!("cargo:warning=CUDA_DEBUG=1 → Enabling comprehensive debugging:");
            println!("cargo:warning=  → CUDA_OPT_LEVEL=0 (no optimization)");
            println!("cargo:warning=  → CUDA_LAUNCH_BLOCKING=1 (synchronous kernels)");
            println!("cargo:warning=  → Line info and device debug symbols enabled");
            println!("cargo:warning=  → CUDA_DEBUG macro defined for preprocessor");
            
            builder.flag("-G");                                    // Device debug symbols
            builder.flag("-Xcompiler=-fno-omit-frame-pointer");    // Better stack traces
            builder.flag("-Xptxas=-v");                            // Verbose PTX compilation
            builder.define("CUDA_DEBUG", "1");            // Define CUDA_DEBUG macro
        }
    }
}

/// Check if CUDA is available on the system
pub fn cuda_available() -> bool {
    Command::new("nvcc").arg("--version").output().is_ok()
}

/// Detect CUDA architecture using nvidia-smi
pub fn detect_cuda_arch() -> String {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .expect("Failed to run nvidia-smi - make sure NVIDIA drivers are installed");

    let full_output =
        String::from_utf8(output.stdout).expect("nvidia-smi output is not valid UTF-8");

    let arch = full_output
        .lines()
        .next()
        .expect("nvidia-smi failed to return compute capability")
        .trim()
        .replace('.', ""); // Convert "7.5" to "75"

    // Set both cargo env and process env
    println!("cargo:rustc-env=CUDA_ARCH={}", arch);
    env::set_var("CUDA_ARCH", &arch);
    arch
}

/// Calculate optimal number of parallel NVCC jobs
pub fn nvcc_parallel_jobs() -> String {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let threads = env::var("NVCC_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(threads);

    format!("-t{}", threads)
}

/// Watch files matching a glob pattern
fn watch_glob(pattern: &str) {
    for path in glob::glob(pattern).expect("Invalid glob pattern").flatten() {
        if path.is_file() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
