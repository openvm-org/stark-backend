#!/usr/bin/env python3
"""Render an overview PNG from a binary aggregate produced by `aggregate`.

Usage:
    python3 render_overview.py <agg.bin> <output.png>

The aggregate is the small (~MB) binary produced by `./aggregate
shadow_profile.bin agg.bin` — it contains a per-(SM, time-bucket) CTA
count grid, a per-(kernel, time-bucket) cumulative-CTA-residency grid,
per-kernel totals, and the kernel name table. This script renders two
panels: a per-bucket-normalized stacked area showing CTA-residency
composition by top-12 kernel, and a per-SM heatmap (rows = SM, cols =
time bucket, color = CTA count landing in that cell).
"""

import struct
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} <agg.bin> <output.png>', file=sys.stderr)
    sys.exit(1)
PATH = sys.argv[1]
OUT = sys.argv[2]


def short(name):
    s = name
    while s and ord(s[0]) < 0x30:
        s = s[1:]
    try:
        import shutil, subprocess
        if shutil.which('c++filt') is not None:
            out = subprocess.run(['c++filt'], input=s, capture_output=True,
                                 text=True, timeout=2).stdout.strip()
            if out:
                out = out.split('(')[0]
                if '::' in out:
                    parts = out.split('::')
                    out = '::'.join(parts[-2:])
                if len(out) > 60:
                    out = out[:60]
                return out
    except Exception:
        pass
    s = s.split('(')[0].split('<')[0].lstrip('_Z123456789NM')
    if len(s) > 60:
        s = s[:60]
    return s.strip() or 'kernel'


with open(PATH, 'rb') as f:
    data = f.read()

# Header: t_min t_max n_cta n_drop  +  n_kernels n_sms n_buckets _pad
t_min, t_max, n_cta, n_drop = struct.unpack_from('<QQQQ', data, 0)
n_kernels, N_SMS, N_BUCKETS, _ = struct.unpack_from('<IIII', data, 32)
print(f't_min={t_min}, t_max={t_max}, n_cta={n_cta:,}, n_drop={n_drop:,}',
      file=sys.stderr)
print(f'n_kernels={n_kernels}, N_SMS={N_SMS}, N_BUCKETS={N_BUCKETS}',
      file=sys.stderr)

off = 48
heat = np.frombuffer(data, dtype=np.uint32, count=N_SMS * N_BUCKETS,
                     offset=off).reshape(N_SMS, N_BUCKETS).astype(np.int64)
off += N_SMS * N_BUCKETS * 4

ktime = np.frombuffer(data, dtype=np.uint64,
                      count=(n_kernels + 1) * N_BUCKETS,
                      offset=off).reshape(n_kernels + 1, N_BUCKETS).astype(np.float64)
off += (n_kernels + 1) * N_BUCKETS * 8

ktotal = np.frombuffer(data, dtype=np.uint64, count=n_kernels + 1,
                       offset=off).copy().astype(np.float64)
off += (n_kernels + 1) * 8

# Kernel name list.
kernels = {}     # kid -> name
kid_by_idx = []  # idx -> kid
for _ in range(n_kernels):
    kid, nlen = struct.unpack_from('<II', data, off)
    off += 8
    name = data[off:off + nlen].decode('utf-8', errors='replace')
    off += nlen
    kernels[kid] = name
    kid_by_idx.append(kid)

duration_s = (t_max - t_min) / 1e9
bucket_ns = (t_max - t_min) / N_BUCKETS

# Per-(kernel, bucket) CTA counts derived from the heatmap is not available
# in the aggregate (we only have it per-SM, not per-kernel). The stacked
# composition view uses `ktime[k, b]` (per-kernel cumulative CTA-residency
# time per bucket) as a *proxy* for "which kernels are firing in this
# bucket" — its absolute scale is meaningless (sum of CTA-residencies, not
# wall-clock time, see SUMMARY.md), but the *relative* shape per bucket
# correctly identifies kernel composition over time. We normalize per-
# bucket so each column sums to 1 and the user can read it as
# "what fraction of CTA-residency in this bucket came from each kernel".
ktotal_real = ktotal[:n_kernels]
top = np.argsort(-ktotal_real)[:12]
top_kid = [kid_by_idx[i] for i in top]
print('top-12 kernels by cumulative CTA-residency (proxy for activity share):',
      file=sys.stderr)
for ti, i in enumerate(top):
    name = short(kernels[kid_by_idx[i]])
    print(f'  {ti}: {ktotal_real[i]/1e9*1000:>10.1f} ms-CTA  {name}',
          file=sys.stderr)

activity = np.zeros((len(top) + 1, N_BUCKETS), dtype=np.float64)
for j, i in enumerate(top):
    activity[j] = ktime[i]
mask = np.ones(n_kernels + 1, dtype=bool)
mask[top] = False
activity[-1] = ktime[mask].sum(axis=0)
# Per-bucket normalization so the stack reads as composition (sums to 1
# wherever any CTA fired, 0 otherwise).
col_sum = activity.sum(axis=0)
nonzero = col_sum > 0
activity_norm = np.zeros_like(activity)
activity_norm[:, nonzero] = activity[:, nonzero] / col_sum[nonzero]

fig = plt.figure(figsize=(22, 11), dpi=110)
gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 4.0], hspace=0.28)

ax_act = fig.add_subplot(gs[0])
ts = np.linspace(0, duration_s, N_BUCKETS)
colors = list(plt.cm.tab20.colors[:len(top)]) + [(0.5, 0.5, 0.5, 1.0)]
labels = [short(kernels[k]) for k in top_kid] + ['(other)']
ax_act.stackplot(ts, activity_norm, labels=labels, colors=colors, edgecolor='none')
ax_act.set_ylim(0, 1)
ax_act.set_xlim(0, duration_s)
ax_act.set_ylabel('CTA-residency share\n(normalized per bucket)')
ax_act.set_xlabel('time (s, since GPU t0)')
ax_act.set_title(
    f'NVBit per-CTA profile — '
    f'{n_cta:,} CTAs over {duration_s:.0f}s, '
    f'{n_kernels} distinct kernel families'
    + (f' — {n_drop:,} drops' if n_drop else ''),
    pad=8,
)
ax_act.legend(loc='center left', bbox_to_anchor=(1.005, 0.5), fontsize=7,
              frameon=False, labelspacing=0.4)

# Per-SM heatmap.
sms_active = np.where(heat.sum(axis=1) > 0)[0]
sm_min = int(sms_active.min()) if len(sms_active) else 0
sm_max = int(sms_active.max()) if len(sms_active) else 1
heat_view = heat[sm_min:sm_max + 1]
ax_heat = fig.add_subplot(gs[1], sharex=ax_act)
norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=max(1, heat_view.max()))
im = ax_heat.imshow(
    heat_view, aspect='auto', interpolation='nearest', cmap='magma',
    extent=[0, duration_s, sm_max + 0.5, sm_min - 0.5], norm=norm,
)
ax_heat.set_ylabel('SM index')
ax_heat.set_xlabel('time (s, since GPU t0)')
ax_heat.set_title(
    f'Per-SM CTA placement (every kernel via NVBit) — '
    f'{sm_max - sm_min + 1} SMs covered — '
    f'color = sqrt of CTAs landing per (SM, {bucket_ns/1e6:.0f} ms bucket)',
    pad=8,
)
cbar = fig.colorbar(im, ax=ax_heat, fraction=0.018, pad=0.01)
cbar.set_label('CTAs per cell')

fig.suptitle(
    'Hardware-mapped CUDA profiler (NVBit, every kernel)',
    fontsize=13, y=0.995,
)
fig.savefig(OUT, dpi=110, bbox_inches='tight')
print(f'wrote {OUT}', file=sys.stderr)
