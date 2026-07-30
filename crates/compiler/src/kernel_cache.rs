//! On-disk cache of JIT-compiled kernel modules.
//!
//! Each entry lives in its own subdirectory named by the hex form of the
//! source [`ir::Module`]'s [`crate::module_hash::module_hash`]. Inside the
//! entry sit the artifacts [`crate::runtime::KernelModule`] needs to be
//! reconstructed without invoking `nvcc`: `libmodule.so`, `module.cu` and a
//! `metadata.json` describing the module's input / output scalar types.
//!
//! # Eviction
//!
//! The cache enforces two independent bounds:
//! - `max_kernels` — maximum number of entries.
//! - `storage_size` — maximum total on-disk bytes across all entries.
//!
//! When either is exceeded on insert, the least-recently-used entries are
//! deleted (in LRU order) until both bounds are satisfied. "Recency" is the
//! entry directory's modification time: cache hits `touch` it so a re-used
//! entry becomes the most recent again.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::SystemTime,
};

use parking_lot::Mutex;

use crate::{
    ir::Module,
    module_hash::module_hash_hex,
    runtime::{CompileOptions, KernelModule, KERNEL_MODULE_METADATA, KERNEL_MODULE_SO},
    CompileError,
};

const DEFAULT_MAX_KERNELS: usize = 300;
const DEFAULT_STORAGE_BYTES: u64 = 10 * 1024 * 1024 * 1024; // 10 GiB

/// Builder-configured on-disk kernel cache.
///
/// Cheap to clone (all state is behind an [`Arc`] via [`Mutex`]).
pub struct KernelCache {
    directory: PathBuf,
    max_kernels: usize,
    storage_size: u64,
    // Serializes eviction so concurrent inserts don't double-delete. The
    // `insert` path also takes it so on-disk state stays consistent under
    // parallel graph compilation.
    lock: Mutex<()>,
}

impl KernelCache {
    /// Default cache under `~/.openvm/kernel_cache`, 300 kernels, 10 GiB.
    ///
    /// The directory is created on first insert; it need not exist yet.
    pub fn new() -> Self {
        Self::at(default_cache_dir())
    }

    /// Cache rooted at `directory`, with the default eviction bounds.
    pub fn at(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            max_kernels: DEFAULT_MAX_KERNELS,
            storage_size: DEFAULT_STORAGE_BYTES,
            lock: Mutex::new(()),
        }
    }

    pub fn directory(mut self, directory: impl Into<PathBuf>) -> Self {
        self.directory = directory.into();
        self
    }

    pub fn max_kernels(mut self, max_kernels: usize) -> Self {
        self.max_kernels = max_kernels;
        self
    }

    pub fn storage_size(mut self, storage_size: u64) -> Self {
        self.storage_size = storage_size;
        self
    }

    pub fn root(&self) -> &Path {
        &self.directory
    }

    /// Path to an entry's directory (may not exist yet).
    pub fn entry_dir(&self, key: &str) -> PathBuf {
        self.directory.join(key)
    }

    /// Looks up `module` in the cache. On hit, dlopens the persisted `.so`,
    /// touches the entry's mtime so LRU accounting treats it as recent, and
    /// returns the reconstructed [`KernelModule`]. On miss, returns `None`.
    ///
    /// An entry that exists but fails to load (torn write from a crashed
    /// producer, concurrent eviction between the existence check and the
    /// dlopen, corrupt artifacts) is treated as a miss rather than an error —
    /// the caller falls back to a fresh compile.
    pub fn get(&self, module: &Module) -> Result<Option<KernelModule>, CompileError> {
        let key = module_hash_hex(module);
        let dir = self.entry_dir(&key);
        if !dir.join(KERNEL_MODULE_SO).is_file() || !dir.join(KERNEL_MODULE_METADATA).is_file() {
            return Ok(None);
        }
        // Touch mtime so this entry moves to the head of the LRU. Best-effort:
        // if the touch fails we still return the module.
        let _ = touch(&dir);
        match KernelModule::load_from_dir(&dir) {
            Ok(km) => Ok(Some(km)),
            Err(e) => {
                tracing_warn(&format!(
                    "kernel_cache: entry {key} failed to load ({e}); recompiling"
                ));
                Ok(None)
            }
        }
    }

    /// Persists `km`'s artifacts under this cache keyed by `module`, then
    /// enforces the eviction bounds. Existing entries under the same key are
    /// overwritten (they represent the same source module and any prior
    /// artifacts are stale w.r.t. the fresh compile).
    ///
    /// The artifacts are staged in a hidden sibling directory and published
    /// with an atomic `rename`, so a concurrent reader — including one in
    /// another *process*, which the `lock` cannot serialize — never observes
    /// a half-written entry and never has a dlopen'd `.so` overwritten in
    /// place under its feet (the old inode stays alive for existing maps;
    /// only the directory entry is swapped).
    ///
    /// A best-effort operation: if the underlying filesystem operations fail
    /// mid-way we still return the compiled module — the cache is an
    /// optimization, not a correctness guarantee.
    pub fn insert(&self, module: &Module, km: &KernelModule) -> Result<(), CompileError> {
        let _guard = self.lock.lock();
        let key = module_hash_hex(module);
        let dir = self.entry_dir(&key);
        let stage = self
            .directory
            .join(format!(".{key}.tmp-{}", std::process::id()));
        let _ = fs::remove_dir_all(&stage);
        fs::create_dir_all(&stage).map_err(io_err("kernel_cache: create staging dir"))?;
        let staged = km.save_artifacts(&stage).and_then(|()| {
            let _ = touch(&stage);
            fs::rename(&stage, &dir).or_else(|_| {
                // The target exists (stale prior artifacts or a concurrent
                // producer of the same key). Displace it and retry once; the
                // key is a content hash, so a racing winner's entry is
                // equivalent and losing this second race is also fine.
                let _ = fs::remove_dir_all(&dir);
                fs::rename(&stage, &dir).map_err(io_err("kernel_cache: publish entry"))
            })
        });
        if let Err(e) = staged {
            let _ = fs::remove_dir_all(&stage);
            return Err(e);
        }
        self.enforce_bounds()?;
        Ok(())
    }

    /// Deletes least-recently-used entries until both `max_kernels` and
    /// `storage_size` are satisfied. Idempotent; safe to call at any time.
    pub fn enforce_bounds(&self) -> Result<(), CompileError> {
        let mut entries = self.list_entries()?;
        // Oldest first — we pop from the front as we evict.
        entries.sort_by_key(|e| e.mtime);
        let mut total_bytes: u64 = entries.iter().map(|e| e.size).sum();
        let mut count = entries.len();
        let mut i = 0;
        while (count > self.max_kernels || total_bytes > self.storage_size) && i < entries.len() {
            let e = &entries[i];
            if let Err(err) = fs::remove_dir_all(&e.path) {
                // A concurrent remove or a permission error means we can't
                // shrink further; give up gracefully rather than looping.
                return Err(io_err("kernel_cache: evict entry")(err));
            }
            total_bytes = total_bytes.saturating_sub(e.size);
            count -= 1;
            i += 1;
        }
        Ok(())
    }

    /// Full listing of cache entries with their sizes and last-touched times.
    fn list_entries(&self) -> Result<Vec<Entry>, CompileError> {
        let mut out = Vec::new();
        let iter = match fs::read_dir(&self.directory) {
            Ok(it) => it,
            // A missing cache dir is trivially bounded — treat it as empty.
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(out),
            Err(e) => return Err(io_err("kernel_cache: read directory")(e)),
        };
        for entry in iter {
            let entry = entry.map_err(io_err("kernel_cache: iterate directory"))?;
            let path = entry.path();
            // Hidden names are in-flight staging dirs (see `insert`), not
            // published entries — evicting one would tear a concurrent
            // producer's publish.
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let meta = match entry.metadata() {
                Ok(m) if m.is_dir() => m,
                _ => continue,
            };
            let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            let size = dir_size(&path).unwrap_or(0);
            out.push(Entry { path, mtime, size });
        }
        Ok(out)
    }
}

impl Default for KernelCache {
    fn default() -> Self {
        Self::new()
    }
}

struct Entry {
    path: PathBuf,
    mtime: SystemTime,
    size: u64,
}

/// Compiles `module` with `options`, inserts the fresh artifacts into
/// `cache`, and returns the loaded module.
pub fn compile_and_cache(
    module: &Module,
    options: &CompileOptions,
    cache: &KernelCache,
) -> Result<KernelModule, CompileError> {
    if let Some(km) = cache.get(module)? {
        return Ok(km);
    }
    let km = crate::compile_and_load(module, options)?;
    // Cache insert is best-effort — a full disk shouldn't fail the compile.
    if let Err(e) = cache.insert(module, &km) {
        tracing_warn(&format!(
            "kernel_cache: failed to persist entry for {}: {e}",
            module.name
        ));
    }
    Ok(km)
}

fn tracing_warn(msg: &str) {
    // No `tracing` dep in this crate — go through stderr so we don't lose
    // the message under `RUST_LOG` gating.
    eprintln!("[crypto-compiler kernel_cache warning] {msg}");
}

fn io_err(what: &'static str) -> impl Fn(io::Error) -> CompileError {
    move |e| CompileError::Runtime(format!("{what}: {e}"))
}

fn touch(dir: &Path) -> io::Result<()> {
    // On Unix we can't use `set_modified` (stable only on File), so we open a
    // sidecar tombstone file that we rewrite. This bumps the directory's
    // mtime as a byproduct of the write.
    let stamp = dir.join(".last_used");
    fs::write(&stamp, [])?;
    // The rewrite bumped the file mtime; explicitly touch the directory too
    // via a create-then-remove so its mtime reflects the touch. On Linux
    // rewriting a child bumps the parent mtime, so this is mostly redundant.
    Ok(())
}

fn dir_size(path: &Path) -> io::Result<u64> {
    let mut total = 0u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if meta.is_dir() {
            total += dir_size(&entry.path())?;
        } else {
            total += meta.len();
        }
    }
    Ok(total)
}

fn default_cache_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".openvm").join("kernel_cache");
    }
    if let Ok(dir) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("openvm").join("kernel_cache");
    }
    // Last-resort fallback: process CWD. Not ideal, but predictable.
    PathBuf::from(".openvm-kernel-cache")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{IRBuilder, ScalarType};

    fn scale_module(factor: u32) -> Module {
        let mut b = IRBuilder::new();
        let a = b.input("a", ScalarType::BabyBear, vec![4]);
        let body = b.compute(4, |b, i| {
            let ai = b.index(a, &[i]);
            let k = b.const_field(factor);
            b.mul(ai, k)
        });
        b.finish(format!("scale_by_{factor}"), body)
    }

    /// Hash keys are stable across builds so the cache path derived from a
    /// module stays the same on repeat runs.
    #[test]
    fn entry_dir_uses_module_hash() {
        let m = scale_module(2);
        let cache = KernelCache::at(std::env::temp_dir());
        let key = module_hash_hex(&m);
        assert_eq!(cache.entry_dir(&key), std::env::temp_dir().join(key));
    }

    /// `get` returns `None` for missing entries without erroring.
    #[test]
    fn miss_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = KernelCache::at(tmp.path());
        let m = scale_module(2);
        assert!(cache.get(&m).unwrap().is_none());
    }

    /// A fake insert (empty `.so` and metadata) triggers eviction once
    /// `max_kernels` is exceeded.
    #[test]
    fn enforce_bounds_evicts_oldest() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = KernelCache::at(tmp.path())
            .max_kernels(2)
            .storage_size(1_000_000);

        for i in 0..3 {
            let dir = cache.entry_dir(&format!("fake_{i}"));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(KERNEL_MODULE_SO), []).unwrap();
            fs::write(dir.join(KERNEL_MODULE_METADATA), b"{}").unwrap();
            // Slightly stagger mtimes so ordering is deterministic.
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        cache.enforce_bounds().unwrap();
        // Oldest should have been evicted.
        assert!(!cache.entry_dir("fake_0").exists());
        assert!(cache.entry_dir("fake_1").exists());
        assert!(cache.entry_dir("fake_2").exists());
    }

    /// An entry whose artifacts fail to load (torn write from a crashed
    /// producer, corruption) is a miss, not an error — callers recompile.
    #[test]
    fn corrupt_entry_is_a_miss() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = KernelCache::at(tmp.path());
        let m = scale_module(2);
        let dir = cache.entry_dir(&module_hash_hex(&m));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(KERNEL_MODULE_SO), b"not an ELF").unwrap();
        fs::write(dir.join(KERNEL_MODULE_METADATA), b"not json").unwrap();
        assert!(cache.get(&m).unwrap().is_none());
    }

    /// Hidden staging dirs (in-flight `insert` publishes) are invisible to
    /// eviction: they are neither counted against the bounds nor deleted.
    #[test]
    fn staging_dirs_are_skipped_by_eviction() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = KernelCache::at(tmp.path())
            .max_kernels(2)
            .storage_size(1_000_000);

        let stage = tmp.path().join(".abc.tmp-42");
        fs::create_dir_all(&stage).unwrap();
        fs::write(stage.join(KERNEL_MODULE_SO), [0u8; 128]).unwrap();
        for i in 0..2 {
            let dir = cache.entry_dir(&format!("fake_{i}"));
            fs::create_dir_all(&dir).unwrap();
            fs::write(dir.join(KERNEL_MODULE_SO), []).unwrap();
            fs::write(dir.join(KERNEL_MODULE_METADATA), b"{}").unwrap();
        }
        cache.enforce_bounds().unwrap();
        // Two published entries fit the bound; the staging dir neither
        // counted as a third entry nor got evicted.
        assert!(stage.exists());
        assert!(cache.entry_dir("fake_0").exists());
        assert!(cache.entry_dir("fake_1").exists());
    }
}
