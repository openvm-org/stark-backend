/* Streaming SHDWPROF aggregator. Reads a binary log produced by the NVBit
 * tool, emits a compact binary aggregate that the renderer
 * (`render_overview.py`) turns into a PNG.
 *
 * Output layout:
 *   header  : t_min:u64 t_max:u64 n_cta:u64 n_drop:u64
 *             n_kernels:u32 N_SMS:u32 N_BUCKETS:u32 _pad:u32
 *   heatmap : N_SMS * N_BUCKETS u32 (CTAs landing per (smid, bucket))
 *   ktime   : (n_kernels+1) * N_BUCKETS u64 ns
 *             (per-kernel per-bucket cumulative CTA-residency time)
 *   ktotal  : (n_kernels+1) u64 ns total per-kernel CTA-residency time
 *   names   : for each kernel, kid:u32 nlen:u32 name[nlen]
 *
 * Two passes: pass 1 finds t_min, t_max, and the kernel-id table; pass 2
 * aggregates into the fixed-size buckets above. Records never go through
 * a Python list — even at 1.9 billion CTAs (76 GB log) the resident
 * working set is ~100 MB.
 *
 * Build: gcc -O3 -o aggregate aggregate.c -lzstd
 * Run:   ./aggregate <log.bin> <agg.bin>
 *
 * Input is auto-detected as zstd-framed (magic 0x28b52ffd) or raw
 * (magic "SHDWPROF") by peeking the first 4 bytes; both pre- and
 * post-zstd-compression logs are accepted.
 */
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <zstd.h>

#define N_SMS 132
#define N_BUCKETS 1200
#define MAX_KERNELS 4096

#define TAG_PROCESS_START 0x0001u
#define TAG_KERNEL_NAME   0x0002u
#define TAG_CTA_RECORD    0x0003u
#define TAG_DROP          0x0004u

/* ---- Reader: transparent raw-or-zstd input -------------------------------
 *
 * Both the file-position tracking (for progress logging) and the tag/payload
 * parsers go through `r_read`. When `g_zstd` is non-NULL we drive a streaming
 * zstd decoder; otherwise we fread directly. `g_input_pos` tracks bytes read
 * from the actual file (compressed bytes for zstd input, raw bytes
 * otherwise).
 */
static FILE *g_fin = NULL;
static ZSTD_DStream *g_zstd = NULL;
static uint64_t g_file_size = 0;
static uint8_t g_zin_buf[1 << 17]; /* 128 KB compressed-input buffer */
static ZSTD_inBuffer g_zin = {g_zin_buf, 0, 0};

static void r_open(const char *path) {
    g_fin = fopen(path, "rb");
    if (!g_fin) { perror("open input"); exit(1); }
    fseeko(g_fin, 0, SEEK_END);
    g_file_size = (uint64_t)ftello(g_fin);
    fseeko(g_fin, 0, SEEK_SET);

    /* Peek the first 4 bytes to choose raw vs zstd. */
    uint8_t magic[4];
    if (fread(magic, 1, 4, g_fin) != 4) {
        fprintf(stderr, "input is shorter than 4 bytes\n");
        exit(1);
    }
    fseeko(g_fin, 0, SEEK_SET);
    /* zstd frame magic: 0x28 0xb5 0x2f 0xfd, little-endian. */
    int is_zstd = (magic[0] == 0x28 && magic[1] == 0xb5 &&
                   magic[2] == 0x2f && magic[3] == 0xfd);
    if (is_zstd) {
        g_zstd = ZSTD_createDStream();
        if (!g_zstd) {
            fprintf(stderr, "ZSTD_createDStream failed\n");
            exit(1);
        }
        ZSTD_initDStream(g_zstd);
        fprintf(stderr, "input: zstd-framed (file=%.1f GB on disk)\n",
                (double)g_file_size / 1e9);
    } else {
        fprintf(stderr, "input: raw SHDWPROF (file=%.1f GB on disk)\n",
                (double)g_file_size / 1e9);
    }
}

/* Read exactly `n` bytes into `dest`. Returns bytes actually delivered (less
 * than n only at EOF). */
static size_t r_read(void *dest, size_t n) {
    if (!g_zstd) {
        return fread(dest, 1, n, g_fin);
    }
    /* Streaming decompress until we have n bytes (or run out of input). */
    ZSTD_outBuffer out = {dest, n, 0};
    while (out.pos < n) {
        if (g_zin.pos >= g_zin.size) {
            size_t r = fread(g_zin_buf, 1, sizeof(g_zin_buf), g_fin);
            if (r == 0) break;
            g_zin.size = r;
            g_zin.pos = 0;
        }
        size_t status = ZSTD_decompressStream(g_zstd, &out, &g_zin);
        if (ZSTD_isError(status)) {
            fprintf(stderr, "zstd decompress error: %s\n",
                    ZSTD_getErrorName(status));
            exit(1);
        }
        if (status == 0 && g_zin.pos == g_zin.size && out.pos < n) {
            /* End of frame and buffered input drained but caller wanted more
             * — try refilling. If fread returns 0, we're at EOF. */
            size_t r = fread(g_zin_buf, 1, sizeof(g_zin_buf), g_fin);
            if (r == 0) break;
            g_zin.size = r;
            g_zin.pos = 0;
        }
    }
    return out.pos;
}

static void r_rewind(void) {
    fseeko(g_fin, 0, SEEK_SET);
    if (g_zstd) {
        /* Reset the streaming decoder so the next pass decodes from the
         * file's start again. */
        ZSTD_DCtx_reset(g_zstd, ZSTD_reset_session_only);
        g_zin.pos = g_zin.size = 0;
    }
}

static double r_input_progress(void) {
    /* Approximate: fraction of compressed input consumed. */
    return (double)ftello(g_fin) / (double)g_file_size;
}

/* ---- Kernel-id table ----------------------------------------------------- */

typedef struct {
    uint32_t kid;
    char *name;
    uint32_t nlen;
} Kernel;

static Kernel kernels[MAX_KERNELS];
static uint32_t n_kernels = 0;
static uint32_t kid2idx[1u << 23]; /* hash table over kid */

static uint32_t kid_lookup(uint32_t kid, int insert) {
    uint32_t h = kid;
    h ^= h >> 16; h *= 0x85ebca6b; h ^= h >> 13; h *= 0xc2b2ae35; h ^= h >> 16;
    h &= (1u << 23) - 1;
    while (1) {
        uint32_t v = kid2idx[h];
        if (v == 0) {
            if (!insert) return UINT32_MAX;
            if (n_kernels >= MAX_KERNELS) {
                fprintf(stderr, "too many kernels\n");
                exit(2);
            }
            kid2idx[h] = n_kernels + 1; /* offset by 1, 0 = empty */
            kernels[n_kernels].kid = kid;
            kernels[n_kernels].name = NULL;
            kernels[n_kernels].nlen = 0;
            return n_kernels++;
        }
        if (kernels[v - 1].kid == kid) return v - 1;
        h = (h + 1) & ((1u << 23) - 1);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    r_open(argv[1]);
    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { perror("open output"); return 1; }

    /* skip 16-byte header */
    char hdr[16];
    if (r_read(hdr, 16) != 16) {
        fprintf(stderr, "short header\n");
        return 1;
    }
    if (memcmp(hdr, "SHDWPROF", 8) != 0) {
        fprintf(stderr, "bad magic in decoded stream\n");
        return 1;
    }

    /* PASS 1: find t_min / t_max / kernel-id table */
    fprintf(stderr, "pass 1: scanning...\n");
    uint64_t t_min = UINT64_MAX, t_max = 0;
    uint64_t n_cta = 0, n_drop_count = 0;
    uint8_t buf[1 << 20]; /* 1 MB scratch */
    uint64_t poll = 0;
    while (1) {
        uint8_t hh[8];
        size_t r = r_read(hh, 8);
        if (r == 0) break;
        if (r < 8) { fprintf(stderr, "short frame head\n"); break; }
        uint32_t tag = hh[0] | (hh[1] << 8) | (hh[2] << 16) | (hh[3] << 24);
        uint32_t plen = hh[4] | (hh[5] << 8) | (hh[6] << 16) | (hh[7] << 24);
        if (plen > sizeof(buf)) { fprintf(stderr, "payload too big %u\n", plen); return 1; }
        if (r_read(buf, plen) != plen) { fprintf(stderr, "short payload\n"); break; }
        if (tag == TAG_CTA_RECORD) {
            uint64_t t_start, t_end;
            memcpy(&t_start, buf + 16, 8);
            memcpy(&t_end,   buf + 24, 8);
            if (t_start < t_min) t_min = t_start;
            if (t_end   > t_max) t_max = t_end;
            n_cta++;
        } else if (tag == TAG_KERNEL_NAME) {
            uint32_t kid, nlen;
            memcpy(&kid,  buf + 0, 4);
            memcpy(&nlen, buf + 4, 4);
            if (nlen > plen - 8) nlen = plen - 8;
            uint32_t idx = kid_lookup(kid, 1);
            if (kernels[idx].name == NULL) {
                kernels[idx].name = (char *)malloc(nlen + 1);
                memcpy(kernels[idx].name, buf + 8, nlen);
                kernels[idx].name[nlen] = 0;
                kernels[idx].nlen = nlen;
            }
        } else if (tag == TAG_DROP) {
            uint64_t cnt;
            memcpy(&cnt, buf + 0, 8);
            n_drop_count += cnt;
        }
        if ((++poll & ((1u << 23) - 1)) == 0) {
            fprintf(stderr, "  pass1: %" PRIu64 " CTAs, %.1f%% of file read\n",
                    n_cta, r_input_progress() * 100.0);
        }
    }
    fprintf(stderr, "pass 1 done: %" PRIu64 " CTAs, %u kernels, "
            "t_min=%" PRIu64 ", t_max=%" PRIu64 ", drops=%" PRIu64 "\n",
            n_cta, n_kernels, t_min, t_max, n_drop_count);

    if (n_cta == 0 || t_max <= t_min) {
        fprintf(stderr, "no CTA records found\n");
        return 1;
    }

    /* PASS 2: rewind + aggregate */
    fprintf(stderr, "pass 2: aggregating into %dx%d buckets\n", N_SMS, N_BUCKETS);
    r_rewind();
    /* skip header again */
    r_read(hdr, 16);
    uint64_t duration = t_max - t_min;
    uint64_t bucket_ns = duration / N_BUCKETS;
    if (bucket_ns == 0) bucket_ns = 1;

    uint32_t *heat = calloc((size_t)N_SMS * N_BUCKETS, sizeof(uint32_t));
    uint64_t *ktime = calloc((size_t)(n_kernels + 1) * N_BUCKETS, sizeof(uint64_t));
    uint64_t *ktotal = calloc((size_t)(n_kernels + 1), sizeof(uint64_t));

    uint64_t n_cta2 = 0;
    poll = 0;
    while (1) {
        uint8_t hh[8];
        size_t r = r_read(hh, 8);
        if (r == 0) break;
        if (r < 8) break;
        uint32_t tag = hh[0] | (hh[1] << 8) | (hh[2] << 16) | (hh[3] << 24);
        uint32_t plen = hh[4] | (hh[5] << 8) | (hh[6] << 16) | (hh[7] << 24);
        if (plen > sizeof(buf)) break;
        if (r_read(buf, plen) != plen) break;
        if (tag == TAG_CTA_RECORD) {
            uint32_t kid;
            uint32_t smid;
            uint64_t t_start, t_end;
            memcpy(&kid,    buf + 0,  4);
            memcpy(&smid,   buf + 4,  4);
            memcpy(&t_start, buf + 16, 8);
            memcpy(&t_end,   buf + 24, 8);
            uint64_t rel = t_start - t_min;
            uint32_t b = (uint32_t)(rel / bucket_ns);
            if (b >= N_BUCKETS) b = N_BUCKETS - 1;
            if (smid < N_SMS) heat[(size_t)smid * N_BUCKETS + b]++;
            uint32_t ki = kid_lookup(kid, 0);
            if (ki == UINT32_MAX) ki = n_kernels;
            uint64_t dur = t_end >= t_start ? t_end - t_start : 0;
            ktime[(size_t)ki * N_BUCKETS + b] += dur;
            ktotal[ki] += dur;
            n_cta2++;
        }
        if ((++poll & ((1u << 23) - 1)) == 0) {
            fprintf(stderr, "  pass2: %" PRIu64 " / %" PRIu64 " CTAs (%.0f%%), %.1f%% read\n",
                    n_cta2, n_cta, (double)n_cta2 / n_cta * 100.0,
                    r_input_progress() * 100.0);
        }
    }
    fprintf(stderr, "pass 2 done: %" PRIu64 " CTAs aggregated\n", n_cta2);

    /* Write output. */
    struct {
        uint64_t t_min, t_max, n_cta, n_drop;
        uint32_t n_kernels, n_sms, n_buckets, _pad;
    } out_hdr = {t_min, t_max, n_cta2, n_drop_count, n_kernels, N_SMS, N_BUCKETS, 0};
    fwrite(&out_hdr, sizeof(out_hdr), 1, fout);
    fwrite(heat, sizeof(uint32_t), (size_t)N_SMS * N_BUCKETS, fout);
    fwrite(ktime, sizeof(uint64_t), (size_t)(n_kernels + 1) * N_BUCKETS, fout);
    fwrite(ktotal, sizeof(uint64_t), n_kernels + 1, fout);
    for (uint32_t i = 0; i < n_kernels; i++) {
        fwrite(&kernels[i].kid, 4, 1, fout);
        fwrite(&kernels[i].nlen, 4, 1, fout);
        if (kernels[i].nlen) fwrite(kernels[i].name, 1, kernels[i].nlen, fout);
    }
    fclose(fout);
    if (g_zstd) ZSTD_freeDStream(g_zstd);
    fclose(g_fin);
    return 0;
}
