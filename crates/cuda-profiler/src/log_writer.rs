//! Append-only binary log writer.
//!
//! All record producers (the CTA-ring drain thread, the CUPTI sidecar) share
//! a single `LogWriter` behind a `Mutex` and write framed records to a
//! `BufWriter<File>`. The file is opened with `O_CREAT | O_TRUNC` at init.
//! Producers tolerate write errors silently after the first one (we trace
//! once and then drop subsequent writes) — losing diagnostic data is always
//! preferable to crashing the prover.
//!
//! ## Endianness
//!
//! The wire format is **little-endian only**. CUDA targets x86_64/aarch64,
//! both little-endian, so this is unrestrictive in practice. A static check
//! below makes a big-endian build fail to compile rather than silently
//! produce a malformed log.

use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

use tracing::warn;

use crate::record::{Record, MAGIC, VERSION_MAJOR, VERSION_MINOR};

#[cfg(target_endian = "big")]
compile_error!("openvm-cuda-profiler log format is little-endian only; CUDA targets are LE.");

const BUF_BYTES: usize = 1 << 20; // 1 MB

pub struct LogWriter {
    inner: BufWriter<File>,
    path: PathBuf,
    /// True once we've reported a write failure; used to suppress repeats.
    poisoned: bool,
}

impl LogWriter {
    /// Open `path` for write (truncating any existing file) and emit the
    /// 16-byte header.
    pub fn create<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let f = File::create(&path)?;
        let mut w = BufWriter::with_capacity(BUF_BYTES, f);

        // Header: magic + version_major + version_minor + reserved.
        w.write_all(MAGIC)?;
        w.write_all(&VERSION_MAJOR.to_le_bytes())?;
        w.write_all(&VERSION_MINOR.to_le_bytes())?;
        w.write_all(&[0u8; 4])?;

        Ok(Self {
            inner: w,
            path,
            poisoned: false,
        })
    }

    /// Write one framed record. Errors are reported once per writer and then
    /// silently dropped, so a full disk doesn't propagate into the prover.
    pub fn write_record(&mut self, rec: &Record) -> io::Result<()> {
        if self.poisoned {
            return Ok(());
        }
        let mut payload = Vec::with_capacity(64);
        rec.encode_payload(&mut payload)?;
        let plen: u32 = payload
            .len()
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "record payload too large"))?;
        let tag = rec.tag() as u32;
        if let Err(e) = self
            .inner
            .write_all(&tag.to_le_bytes())
            .and_then(|_| self.inner.write_all(&plen.to_le_bytes()))
            .and_then(|_| self.inner.write_all(&payload))
        {
            warn!(path = %self.path.display(), error = %e, "cuda-profiler: log write failed");
            self.poisoned = true;
            return Err(e);
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        if self.poisoned {
            return Ok(());
        }
        self.inner.flush()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
