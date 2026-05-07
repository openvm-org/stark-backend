//! Binary log record types and on-disk format.
//!
//! The log is a length-prefixed stream of records. The format is intentionally
//! simple — not protobuf — so the assembler can stream-decode without pulling
//! in any new dependency, and the format is stable enough to checkpoint a
//! profile run and inspect it later.
//!
//! ```text
//! [16-byte header]
//!   "SHDWPROF" (8 ASCII bytes)
//!   u16 version_major   = 1
//!   u16 version_minor   = 0
//!   u32 reserved        = 0
//! [records...]
//!   u32 type_tag (LE)
//!   u32 payload_len (LE)
//!   <payload_len bytes>
//! ```
//!
//! Numeric fields are little-endian; CTA records are written as the raw 24-byte
//! struct. String payloads are length-prefixed by the outer record-length.

use std::io::{self, Read, Write};

use crate::ffi::CtaRecord;

pub const MAGIC: &[u8; 8] = b"SHDWPROF";
pub const VERSION_MAJOR: u16 = 1;
pub const VERSION_MINOR: u16 = 0;

/// Type tags. Reserve a u32 so we can grow without recompiling old assemblers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    /// Process info: `pid:u32, gpu_index:u32, start_walltime_ns:u64, gpu_name:str`.
    ProcessStart = 0x0001,
    /// Mapping: `id:u32, name:str`.
    KernelName = 0x0002,
    /// CTA telemetry from the device-side ring: raw `CtaRecord` (24 bytes).
    CtaRecord = 0x0003,
    /// Producer lapped consumer: `count:u64, head_at_detection:u64,
    /// approx_t_ns:u64`. `approx_t_ns` is the most recent CTA timestamp
    /// observed before the overrun, so the assembler can place the drop
    /// marker on the timeline (otherwise it would anchor at t=0).
    Drop = 0x0004,
    /// CUPTI Activity kernel record: `correlation_id:u32, stream_id:u32,
    /// device_id:u32, t_start_ns:u64, t_end_ns:u64, name:str`.
    CuptiKernel = 0x0010,
    /// CUPTI Activity memcpy record: `kind:u32, src_kind:u32, dst_kind:u32,
    /// bytes:u64, t_start_ns:u64, t_end_ns:u64, stream_id:u32`.
    CuptiMemcpy = 0x0011,
    /// CUPTI Activity NVTX marker. `kind` is 1=instantaneous, 2=start, 4=end
    /// (matches `CUPTI_ACTIVITY_FLAG_MARKER_*`). The assembler pairs
    /// START/END records by `(domain, id)`. END markers carry an empty name
    /// per the CUPTI ABI; the assembler reuses the matching START's name.
    CuptiNvtxRange = 0x0012,
    /// CUPTI Activity device runtime overhead (CTX/sync) record:
    /// `kind:u32, t_start_ns:u64, t_end_ns:u64`.
    CuptiOverhead = 0x0013,
    /// One line of `runtime_choices.jsonl`, captured raw for the assembler.
    /// `t_walltime_ns:u64, json:str`.
    RuntimeChoice = 0x0020,
    /// PC sampling stall reason: `t_ns:u64, kernel_correlation:u32,
    /// stall_reason:u32, count:u32`.
    PcSample = 0x0030,
    /// Range Profiler sample: `range_name:str, metric_name:str, value:f64`.
    RangeMetric = 0x0040,
}

#[derive(Debug, Clone)]
pub enum Record {
    ProcessStart {
        pid: u32,
        gpu_index: u32,
        start_walltime_ns: u64,
        gpu_name: String,
    },
    KernelName {
        id: u32,
        name: String,
    },
    Cta(CtaRecord),
    Drop {
        count: u64,
        head_at_detection: u64,
        approx_t_ns: u64,
    },
    CuptiKernel {
        correlation_id: u32,
        stream_id: u32,
        device_id: u32,
        t_start_ns: u64,
        t_end_ns: u64,
        name: String,
    },
    CuptiMemcpy {
        copy_kind: u32,
        src_kind: u32,
        dst_kind: u32,
        bytes: u64,
        t_start_ns: u64,
        t_end_ns: u64,
        stream_id: u32,
    },
    CuptiNvtxRange {
        domain: String,
        /// Marker name. Empty on END records (CUPTI doesn't provide it).
        name: String,
        /// Marker timestamp in ns.
        timestamp_ns: u64,
        /// Per-`(domain, id)` pairing key.
        id: u32,
        /// `CUPTI_ACTIVITY_FLAG_MARKER_*`: 1=instant, 2=start, 4=end.
        marker_kind: u32,
    },
    CuptiOverhead {
        kind: u32,
        t_start_ns: u64,
        t_end_ns: u64,
    },
    RuntimeChoice {
        t_walltime_ns: u64,
        json: String,
    },
    PcSample {
        t_ns: u64,
        kernel_correlation: u32,
        stall_reason: u32,
        count: u32,
    },
    RangeMetric {
        range_name: String,
        metric_name: String,
        value: f64,
    },
}

impl Record {
    pub fn tag(&self) -> Tag {
        match self {
            Record::ProcessStart { .. } => Tag::ProcessStart,
            Record::KernelName { .. } => Tag::KernelName,
            Record::Cta(_) => Tag::CtaRecord,
            Record::Drop { .. } => Tag::Drop,
            Record::CuptiKernel { .. } => Tag::CuptiKernel,
            Record::CuptiMemcpy { .. } => Tag::CuptiMemcpy,
            Record::CuptiNvtxRange { .. } => Tag::CuptiNvtxRange,
            Record::CuptiOverhead { .. } => Tag::CuptiOverhead,
            Record::RuntimeChoice { .. } => Tag::RuntimeChoice,
            Record::PcSample { .. } => Tag::PcSample,
            Record::RangeMetric { .. } => Tag::RangeMetric,
        }
    }

    /// Encode payload (without the 8-byte tag/length frame).
    pub fn encode_payload<W: Write>(&self, w: &mut W) -> io::Result<()> {
        match self {
            Record::ProcessStart {
                pid,
                gpu_index,
                start_walltime_ns,
                gpu_name,
            } => {
                w.write_all(&pid.to_le_bytes())?;
                w.write_all(&gpu_index.to_le_bytes())?;
                w.write_all(&start_walltime_ns.to_le_bytes())?;
                write_str(w, gpu_name)?;
            }
            Record::KernelName { id, name } => {
                w.write_all(&id.to_le_bytes())?;
                write_str(w, name)?;
            }
            Record::Cta(rec) => {
                w.write_all(&rec.kernel_id.to_le_bytes())?;
                w.write_all(&rec.smid.to_le_bytes())?;
                w.write_all(&rec.block_linear.to_le_bytes())?;
                w.write_all(&rec.seq_tag.to_le_bytes())?;
                w.write_all(&rec.t_start.to_le_bytes())?;
                w.write_all(&rec.t_end.to_le_bytes())?;
            }
            Record::Drop {
                count,
                head_at_detection,
                approx_t_ns,
            } => {
                w.write_all(&count.to_le_bytes())?;
                w.write_all(&head_at_detection.to_le_bytes())?;
                w.write_all(&approx_t_ns.to_le_bytes())?;
            }
            Record::CuptiKernel {
                correlation_id,
                stream_id,
                device_id,
                t_start_ns,
                t_end_ns,
                name,
            } => {
                w.write_all(&correlation_id.to_le_bytes())?;
                w.write_all(&stream_id.to_le_bytes())?;
                w.write_all(&device_id.to_le_bytes())?;
                w.write_all(&t_start_ns.to_le_bytes())?;
                w.write_all(&t_end_ns.to_le_bytes())?;
                write_str(w, name)?;
            }
            Record::CuptiMemcpy {
                copy_kind,
                src_kind,
                dst_kind,
                bytes,
                t_start_ns,
                t_end_ns,
                stream_id,
            } => {
                w.write_all(&copy_kind.to_le_bytes())?;
                w.write_all(&src_kind.to_le_bytes())?;
                w.write_all(&dst_kind.to_le_bytes())?;
                w.write_all(&bytes.to_le_bytes())?;
                w.write_all(&t_start_ns.to_le_bytes())?;
                w.write_all(&t_end_ns.to_le_bytes())?;
                w.write_all(&stream_id.to_le_bytes())?;
            }
            Record::CuptiNvtxRange {
                domain,
                name,
                timestamp_ns,
                id,
                marker_kind,
            } => {
                write_str(w, domain)?;
                write_str(w, name)?;
                w.write_all(&timestamp_ns.to_le_bytes())?;
                w.write_all(&id.to_le_bytes())?;
                w.write_all(&marker_kind.to_le_bytes())?;
            }
            Record::CuptiOverhead {
                kind,
                t_start_ns,
                t_end_ns,
            } => {
                w.write_all(&kind.to_le_bytes())?;
                w.write_all(&t_start_ns.to_le_bytes())?;
                w.write_all(&t_end_ns.to_le_bytes())?;
            }
            Record::RuntimeChoice {
                t_walltime_ns,
                json,
            } => {
                w.write_all(&t_walltime_ns.to_le_bytes())?;
                write_str(w, json)?;
            }
            Record::PcSample {
                t_ns,
                kernel_correlation,
                stall_reason,
                count,
            } => {
                w.write_all(&t_ns.to_le_bytes())?;
                w.write_all(&kernel_correlation.to_le_bytes())?;
                w.write_all(&stall_reason.to_le_bytes())?;
                w.write_all(&count.to_le_bytes())?;
            }
            Record::RangeMetric {
                range_name,
                metric_name,
                value,
            } => {
                write_str(w, range_name)?;
                write_str(w, metric_name)?;
                w.write_all(&value.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

fn write_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    let len: u32 = bytes
        .len()
        .try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "string too long"))?;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(bytes)
}

fn read_str<R: Read>(r: &mut R) -> io::Result<String> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 1 << 24 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("string length {len} exceeds 16 MB sanity cap"),
        ));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Read one (tag, payload-bytes) pair from `r`. Returns `None` on EOF.
pub fn read_frame<R: Read>(r: &mut R) -> io::Result<Option<(u32, Vec<u8>)>> {
    let mut head = [0u8; 8];
    match r.read_exact(&mut head) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let tag = u32::from_le_bytes([head[0], head[1], head[2], head[3]]);
    let plen = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
    if plen > 64 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("record payload {plen} exceeds 64 MB sanity cap"),
        ));
    }
    let mut payload = vec![0u8; plen];
    r.read_exact(&mut payload)?;
    Ok(Some((tag, payload)))
}

/// Decode a payload back into a [`Record`] given its tag. Mirrors `encode_payload`.
pub fn decode_payload(tag: u32, payload: &[u8]) -> io::Result<Record> {
    let mut cur = std::io::Cursor::new(payload);
    if tag == Tag::ProcessStart as u32 {
        let pid = read_u32(&mut cur)?;
        let gpu_index = read_u32(&mut cur)?;
        let start_walltime_ns = read_u64(&mut cur)?;
        let gpu_name = read_str(&mut cur)?;
        Ok(Record::ProcessStart {
            pid,
            gpu_index,
            start_walltime_ns,
            gpu_name,
        })
    } else if tag == Tag::KernelName as u32 {
        let id = read_u32(&mut cur)?;
        let name = read_str(&mut cur)?;
        Ok(Record::KernelName { id, name })
    } else if tag == Tag::CtaRecord as u32 {
        let kernel_id = read_u32(&mut cur)?;
        let smid = read_u32(&mut cur)?;
        let block_linear = read_u32(&mut cur)?;
        let seq_tag = read_u32(&mut cur)?;
        let t_start = read_u64(&mut cur)?;
        let t_end = read_u64(&mut cur)?;
        Ok(Record::Cta(CtaRecord {
            kernel_id,
            smid,
            block_linear,
            seq_tag,
            t_start,
            t_end,
        }))
    } else if tag == Tag::Drop as u32 {
        let count = read_u64(&mut cur)?;
        let head_at_detection = read_u64(&mut cur)?;
        let approx_t_ns = read_u64(&mut cur)?;
        Ok(Record::Drop {
            count,
            head_at_detection,
            approx_t_ns,
        })
    } else if tag == Tag::CuptiKernel as u32 {
        let correlation_id = read_u32(&mut cur)?;
        let stream_id = read_u32(&mut cur)?;
        let device_id = read_u32(&mut cur)?;
        let t_start_ns = read_u64(&mut cur)?;
        let t_end_ns = read_u64(&mut cur)?;
        let name = read_str(&mut cur)?;
        Ok(Record::CuptiKernel {
            correlation_id,
            stream_id,
            device_id,
            t_start_ns,
            t_end_ns,
            name,
        })
    } else if tag == Tag::CuptiMemcpy as u32 {
        let copy_kind = read_u32(&mut cur)?;
        let src_kind = read_u32(&mut cur)?;
        let dst_kind = read_u32(&mut cur)?;
        let bytes = read_u64(&mut cur)?;
        let t_start_ns = read_u64(&mut cur)?;
        let t_end_ns = read_u64(&mut cur)?;
        let stream_id = read_u32(&mut cur)?;
        Ok(Record::CuptiMemcpy {
            copy_kind,
            src_kind,
            dst_kind,
            bytes,
            t_start_ns,
            t_end_ns,
            stream_id,
        })
    } else if tag == Tag::CuptiNvtxRange as u32 {
        let domain = read_str(&mut cur)?;
        let name = read_str(&mut cur)?;
        let timestamp_ns = read_u64(&mut cur)?;
        let id = read_u32(&mut cur)?;
        let marker_kind = read_u32(&mut cur)?;
        Ok(Record::CuptiNvtxRange {
            domain,
            name,
            timestamp_ns,
            id,
            marker_kind,
        })
    } else if tag == Tag::CuptiOverhead as u32 {
        let kind = read_u32(&mut cur)?;
        let t_start_ns = read_u64(&mut cur)?;
        let t_end_ns = read_u64(&mut cur)?;
        Ok(Record::CuptiOverhead {
            kind,
            t_start_ns,
            t_end_ns,
        })
    } else if tag == Tag::RuntimeChoice as u32 {
        let t_walltime_ns = read_u64(&mut cur)?;
        let json = read_str(&mut cur)?;
        Ok(Record::RuntimeChoice {
            t_walltime_ns,
            json,
        })
    } else if tag == Tag::PcSample as u32 {
        let t_ns = read_u64(&mut cur)?;
        let kernel_correlation = read_u32(&mut cur)?;
        let stall_reason = read_u32(&mut cur)?;
        let count = read_u32(&mut cur)?;
        Ok(Record::PcSample {
            t_ns,
            kernel_correlation,
            stall_reason,
            count,
        })
    } else if tag == Tag::RangeMetric as u32 {
        let range_name = read_str(&mut cur)?;
        let metric_name = read_str(&mut cur)?;
        let value = read_f64(&mut cur)?;
        Ok(Record::RangeMetric {
            range_name,
            metric_name,
            value,
        })
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown record tag 0x{tag:04x}"),
        ))
    }
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_kernel_name() {
        let mut buf = Vec::new();
        let rec = Record::KernelName {
            id: 42,
            name: "vslp_round0".into(),
        };
        rec.encode_payload(&mut buf).unwrap();
        let decoded = decode_payload(rec.tag() as u32, &buf).unwrap();
        match decoded {
            Record::KernelName { id, name } => {
                assert_eq!(id, 42);
                assert_eq!(name, "vslp_round0");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn roundtrip_cta() {
        let cta = CtaRecord {
            kernel_id: 7,
            smid: 17,
            block_linear: 1234,
            seq_tag: 0xdead_beef,
            t_start: 1_000_000,
            t_end: 1_500_000,
        };
        let rec = Record::Cta(cta);
        let mut buf = Vec::new();
        rec.encode_payload(&mut buf).unwrap();
        let decoded = decode_payload(rec.tag() as u32, &buf).unwrap();
        match decoded {
            Record::Cta(d) => {
                assert_eq!(d.kernel_id, cta.kernel_id);
                assert_eq!(d.smid, cta.smid);
                assert_eq!(d.seq_tag, cta.seq_tag);
                assert_eq!(d.t_start, cta.t_start);
                assert_eq!(d.t_end, cta.t_end);
            }
            _ => panic!(),
        }
    }
}
