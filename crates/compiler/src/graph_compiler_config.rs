//! TOML-serializable configuration for [`crate::graph_compiler::GraphCompiler`].
//!
//! [`GraphCompilerConfig`] mirrors the builder-style setters on
//! [`GraphCompiler`](crate::graph_compiler::GraphCompiler) so a caller can persist
//! a full tuning profile to a file and rebuild the compiler with
//! [`GraphCompiler::from_toml`](crate::graph_compiler::GraphCompiler::from_toml)
//! or [`GraphCompiler::from_config`](crate::graph_compiler::GraphCompiler::from_config).
//!
//! Fields excluded from the TOML surface (still available on the builder):
//! - Symbol bindings ([`GraphCompiler::symbol`](crate::graph_compiler::GraphCompiler::symbol)):
//!   graph-specific and set by the caller that owns the graph.
//! - Fusion [`estimator`](crate::passes::fusion::FusionOptions::estimator),
//!   [`artifact`](crate::passes::fusion::FusionOptions::artifact), and
//!   [`graph_symbols`](crate::passes::fusion::FusionOptions::graph_symbols): hardware /
//!   graph-specific; defaults are kept.
//! - List-scheduler [`node_times`](crate::planner::ListSchedulerV1::node_times) /
//!   [`node_times`](crate::planner::ListSchedulerV2::node_times): populated from profiling, not
//!   user tuning.

use std::{path::PathBuf, sync::Arc, time::Duration};

use serde::{Deserialize, Serialize};

use crate::{
    graph_ir::DeviceType,
    kernel_cache::KernelCache,
    passes::fusion::FusionOptions,
    planner::{ListSchedulerV1, ListSchedulerV2, SchedulerMode},
    runtime::Verbosity,
};

/// Root TOML config for a [`GraphCompiler`](crate::graph_compiler::GraphCompiler).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GraphCompilerConfig {
    pub device: DeviceConfig,
    pub module_compiler: ModuleCompilerConfig,
    pub scheduler: SchedulerConfig,
    pub kernel_cache: KernelCacheConfig,
    pub fusion: FusionConfig,
}

impl GraphCompilerConfig {
    /// Renders this config as a TOML string.
    pub fn to_toml_string(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Parses a `GraphCompilerConfig` from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }
}

// ---------- Device ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DeviceConfig {
    Cuda { ordinal: usize },
    CpuPinned,
    CpuPaged,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        DeviceConfig::Cuda { ordinal: 0 }
    }
}

impl From<DeviceConfig> for DeviceType {
    fn from(cfg: DeviceConfig) -> Self {
        match cfg {
            DeviceConfig::Cuda { ordinal } => DeviceType::Cuda(ordinal),
            DeviceConfig::CpuPinned => DeviceType::CpuPinned,
            DeviceConfig::CpuPaged => DeviceType::CpuPaged,
        }
    }
}

impl From<DeviceType> for DeviceConfig {
    fn from(d: DeviceType) -> Self {
        match d {
            DeviceType::Cuda(ordinal) => DeviceConfig::Cuda { ordinal },
            DeviceType::CpuPinned => DeviceConfig::CpuPinned,
            DeviceType::CpuPaged => DeviceConfig::CpuPaged,
        }
    }
}

// ---------- ModuleCompiler passthrough ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModuleCompilerConfig {
    pub arch: String,
    pub nvcc: String,
    pub extra_nvcc_flags: Vec<String>,
    pub dump_dir: Option<PathBuf>,
    pub verbosity: Verbosity,
    pub check_accesses: bool,
    /// `None` disables the timeout (unbounded wait).
    pub nvcc_timeout_secs: Option<u64>,
}

impl Default for ModuleCompilerConfig {
    fn default() -> Self {
        let mc = crate::module_compiler::ModuleCompiler::default();
        Self {
            arch: mc.arch,
            nvcc: mc.nvcc,
            extra_nvcc_flags: mc.extra_nvcc_flags,
            dump_dir: mc.dump_dir,
            verbosity: mc.verbosity,
            check_accesses: mc.check_accesses,
            nvcc_timeout_secs: mc.nvcc_timeout.map(|d| d.as_secs()),
        }
    }
}

impl ModuleCompilerConfig {
    pub(crate) fn nvcc_timeout(&self) -> Option<Duration> {
        self.nvcc_timeout_secs.map(Duration::from_secs)
    }
}

// ---------- Scheduler ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum SchedulerConfig {
    ListV1(ListSchedulerV1Config),
    ListV2(ListSchedulerV2Config),
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig::ListV1(ListSchedulerV1Config::default())
    }
}

impl From<SchedulerConfig> for SchedulerMode {
    fn from(cfg: SchedulerConfig) -> Self {
        match cfg {
            SchedulerConfig::ListV1(v) => SchedulerMode::ListV1 { params: v.into() },
            SchedulerConfig::ListV2(v) => SchedulerMode::ListV2 { params: v.into() },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ListSchedulerV1Config {
    pub max_concurrency: u32,
    /// `None` means "no cap" (matches [`ListSchedulerV1::max_memory`] = `u64::MAX`).
    pub max_memory_bytes: Option<u64>,
    pub lookahead_k: usize,
    pub beam: usize,
    pub w_cp: f64,
    pub w_mem: f64,
    pub mem_target_frac: f64,
}

impl Default for ListSchedulerV1Config {
    fn default() -> Self {
        let d = ListSchedulerV1::default();
        Self {
            max_concurrency: d.max_concurrency,
            max_memory_bytes: (d.max_memory != u64::MAX).then_some(d.max_memory),
            lookahead_k: d.lookahead_k,
            beam: d.beam,
            w_cp: d.w_cp,
            w_mem: d.w_mem,
            mem_target_frac: d.mem_target_frac,
        }
    }
}

impl From<ListSchedulerV1Config> for ListSchedulerV1 {
    fn from(c: ListSchedulerV1Config) -> Self {
        ListSchedulerV1 {
            max_concurrency: c.max_concurrency,
            max_memory: c.max_memory_bytes.unwrap_or(u64::MAX),
            lookahead_k: c.lookahead_k,
            beam: c.beam,
            w_cp: c.w_cp,
            w_mem: c.w_mem,
            mem_target_frac: c.mem_target_frac,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ListSchedulerV2Config {
    pub num_streams: usize,
    pub max_memory_bound: usize,
    pub num_beams: usize,
    pub beam_depth: usize,
    pub frontier_cap: usize,
    pub w_m: f64,
    pub w_t: f64,
    pub w_c: f64,
}

impl Default for ListSchedulerV2Config {
    fn default() -> Self {
        let d = ListSchedulerV2::default();
        Self {
            num_streams: d.num_streams,
            max_memory_bound: d.max_memory_bound,
            num_beams: d.num_beams,
            beam_depth: d.beam_depth,
            frontier_cap: d.frontier_cap,
            w_m: d.w_m,
            w_t: d.w_t,
            w_c: d.w_c,
        }
    }
}

impl From<ListSchedulerV2Config> for ListSchedulerV2 {
    fn from(c: ListSchedulerV2Config) -> Self {
        ListSchedulerV2 {
            num_streams: c.num_streams,
            max_memory_bound: c.max_memory_bound,
            num_beams: c.num_beams,
            beam_depth: c.beam_depth,
            frontier_cap: c.frontier_cap,
            w_m: c.w_m,
            w_t: c.w_t,
            w_c: c.w_c,
        }
    }
}

// ---------- Kernel cache ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum KernelCacheConfig {
    Disabled,
    Enabled(KernelCacheEnabledConfig),
}

impl Default for KernelCacheConfig {
    fn default() -> Self {
        KernelCacheConfig::Enabled(KernelCacheEnabledConfig::default())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KernelCacheEnabledConfig {
    /// `None` uses `KernelCache::new()`'s platform default
    /// (`$HOME/.openvm/kernel_cache`, then `$XDG_CACHE_HOME/openvm/kernel_cache`,
    /// then `./.openvm-kernel-cache`).
    pub directory: Option<PathBuf>,
    pub max_kernels: usize,
    pub max_bytes: u64,
}

impl Default for KernelCacheEnabledConfig {
    fn default() -> Self {
        // Mirror the `KernelCache::new()` defaults documented on that type
        // (300 kernels / 10 GiB). Duplicated as literals here so the TOML
        // stays stable when the crate default drifts.
        Self {
            directory: None,
            max_kernels: 300,
            max_bytes: 10 * 1024 * 1024 * 1024,
        }
    }
}

impl KernelCacheEnabledConfig {
    pub(crate) fn build(&self) -> KernelCache {
        let base = match &self.directory {
            Some(dir) => KernelCache::at(dir.clone()),
            None => KernelCache::new(),
        };
        base.max_kernels(self.max_kernels)
            .storage_size(self.max_bytes)
    }
}

impl KernelCacheConfig {
    pub(crate) fn build(&self) -> Option<Arc<KernelCache>> {
        match self {
            KernelCacheConfig::Disabled => None,
            KernelCacheConfig::Enabled(cfg) => Some(Arc::new(cfg.build())),
        }
    }
}

// ---------- Fusion ----------

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FusionConfig {
    Off,
    On(FusionSettings),
}

impl Default for FusionConfig {
    fn default() -> Self {
        FusionConfig::On(FusionSettings::default())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FusionSettings {
    pub max_total_alternatives: usize,
    pub max_enumeration_time_per_round_ms: u64,
    pub max_outer_iterations: usize,
    pub validate_alt_graph_acyclicity: bool,
    pub solver_time_limit_secs: f64,
    pub solver_num_workers: usize,
    pub enable_producer_consumer: bool,
    pub enable_fanout: bool,
    pub enable_small_kernel: bool,
    pub small_kernel_shared_bytes: usize,
    pub small_kernel_max_chain: usize,
    pub enable_horizontal: bool,
    pub enable_epilogue: bool,
    pub enable_keep_variants: bool,
    pub enable_all_keep_variants: bool,
    pub cycle_quantum: i64,
    pub blackbox_hint_cycles: f64,
    pub max_rounds: usize,
    pub max_alternatives_per_pass_per_round: usize,
    pub optimize_artifact_count: bool,
    pub verbose: bool,
}

impl Default for FusionSettings {
    fn default() -> Self {
        let d = FusionOptions::default();
        Self {
            max_total_alternatives: d.max_total_alternatives,
            max_enumeration_time_per_round_ms: d.max_enumeration_time_per_round.as_millis() as u64,
            max_outer_iterations: d.max_outer_iterations,
            validate_alt_graph_acyclicity: d.validate_alt_graph_acyclicity,
            solver_time_limit_secs: d.solver_time_limit_secs,
            solver_num_workers: d.solver_num_workers,
            enable_producer_consumer: d.enable_producer_consumer,
            enable_fanout: d.enable_fanout,
            enable_small_kernel: d.enable_small_kernel,
            small_kernel_shared_bytes: d.small_kernel_shared_bytes,
            small_kernel_max_chain: d.small_kernel_max_chain,
            enable_horizontal: d.enable_horizontal,
            enable_epilogue: d.enable_epilogue,
            enable_keep_variants: d.enable_keep_variants,
            enable_all_keep_variants: d.enable_all_keep_variants,
            cycle_quantum: d.cycle_quantum,
            blackbox_hint_cycles: d.blackbox_hint_cycles,
            max_rounds: d.max_rounds,
            max_alternatives_per_pass_per_round: d.max_alternatives_per_pass_per_round,
            optimize_artifact_count: d.optimize_artifact_count,
            verbose: d.verbose,
        }
    }
}

impl FusionSettings {
    /// Applies TOML-configurable knobs onto a fresh [`FusionOptions`],
    /// leaving fields that are outside the TOML surface (`estimator`,
    /// `artifact`, `graph_symbols`) at their defaults.
    pub(crate) fn to_options(&self) -> FusionOptions {
        FusionOptions {
            max_total_alternatives: self.max_total_alternatives,
            max_enumeration_time_per_round: Duration::from_millis(
                self.max_enumeration_time_per_round_ms,
            ),
            max_outer_iterations: self.max_outer_iterations,
            validate_alt_graph_acyclicity: self.validate_alt_graph_acyclicity,
            solver_time_limit_secs: self.solver_time_limit_secs,
            solver_num_workers: self.solver_num_workers,
            enable_producer_consumer: self.enable_producer_consumer,
            enable_fanout: self.enable_fanout,
            enable_small_kernel: self.enable_small_kernel,
            small_kernel_shared_bytes: self.small_kernel_shared_bytes,
            small_kernel_max_chain: self.small_kernel_max_chain,
            enable_horizontal: self.enable_horizontal,
            enable_epilogue: self.enable_epilogue,
            enable_keep_variants: self.enable_keep_variants,
            enable_all_keep_variants: self.enable_all_keep_variants,
            cycle_quantum: self.cycle_quantum,
            blackbox_hint_cycles: self.blackbox_hint_cycles,
            max_rounds: self.max_rounds,
            max_alternatives_per_pass_per_round: self.max_alternatives_per_pass_per_round,
            optimize_artifact_count: self.optimize_artifact_count,
            verbose: self.verbose,
            ..FusionOptions::default()
        }
    }
}

/// Errors returned by [`GraphCompiler::from_toml`](crate::graph_compiler::GraphCompiler::from_toml).
#[derive(thiserror::Error, Debug)]
pub enum ConfigError {
    #[error("read config {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse config {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_roundtrips_through_toml() {
        let cfg = GraphCompilerConfig::default();
        let s = cfg.to_toml_string().unwrap();
        let back = GraphCompilerConfig::from_toml_str(&s).unwrap();
        // Spot-check every top-level section survives the round trip.
        assert!(matches!(back.device, DeviceConfig::Cuda { ordinal: 0 }));
        assert!(matches!(back.scheduler, SchedulerConfig::ListV1(_)));
        assert!(matches!(back.fusion, FusionConfig::On(_)));
        assert!(matches!(back.kernel_cache, KernelCacheConfig::Enabled(_)));
    }

    #[test]
    fn empty_toml_uses_defaults() {
        let cfg = GraphCompilerConfig::from_toml_str("").unwrap();
        assert!(matches!(cfg.device, DeviceConfig::Cuda { ordinal: 0 }));
    }

    #[test]
    fn scheduler_v2_deserializes() {
        let toml = r#"
            [scheduler]
            mode = "list_v2"
            num_streams = 4
        "#;
        let cfg = GraphCompilerConfig::from_toml_str(toml).unwrap();
        match cfg.scheduler {
            SchedulerConfig::ListV2(v) => assert_eq!(v.num_streams, 4),
            other => panic!("expected ListV2, got {other:?}"),
        }
    }

    #[test]
    #[ignore = "prints the default TOML; run with --nocapture to see"]
    fn print_default_toml() {
        let cfg = GraphCompilerConfig::default();
        // Override env-derived defaults so the printed profile is stable
        // regardless of who runs it (`NVCC`, `CRYPTO_COMPILER_CUDA_ARCH`,
        // `CRYPTO_COMPILER_VERBOSITY`, etc.).
        let cfg = GraphCompilerConfig {
            module_compiler: ModuleCompilerConfig {
                arch: "native".into(),
                nvcc: "nvcc".into(),
                extra_nvcc_flags: Vec::new(),
                dump_dir: None,
                verbosity: Verbosity::Basic,
                check_accesses: false,
                nvcc_timeout_secs: None,
            },
            ..cfg
        };
        println!("{}", cfg.to_toml_string().unwrap());
    }

    #[test]
    fn shipped_default_toml_parses() {
        // Guard against drift: the file living next to Cargo.toml must
        // stay a valid `GraphCompilerConfig`. `CARGO_MANIFEST_DIR` is set
        // by cargo for tests, so this path works regardless of CWD.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cc_default_config.toml");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let cfg = GraphCompilerConfig::from_toml_str(&text)
            .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
        assert!(
            matches!(cfg.fusion, FusionConfig::On(_)),
            "cc_default_config.toml must enable fusion by default; got {:?}",
            cfg.fusion,
        );
    }

    #[test]
    fn shipped_default_toml_builds_compiler() {
        // End-to-end sanity check: `GraphCompiler::from_toml` on the
        // shipped file must succeed. Guards against the config type
        // drifting away from what the builder actually accepts.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cc_default_config.toml");
        crate::graph_compiler::GraphCompiler::from_toml(&path)
            .unwrap_or_else(|e| panic!("from_toml({}): {e}", path.display()));
    }

    #[test]
    fn fusion_off_disables_fusion() {
        let toml = r#"
            [fusion]
            kind = "off"
        "#;
        let cfg = GraphCompilerConfig::from_toml_str(toml).unwrap();
        assert!(matches!(cfg.fusion, FusionConfig::Off));
    }
}
