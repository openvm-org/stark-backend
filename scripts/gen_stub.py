import re, sys, pathlib

hdr_path = sys.argv[1]
hdr = pathlib.Path(hdr_path).read_text()

# Match extern function declarations: "extern <ret> name(...);"
pat = re.compile(r'^extern\s[^;{]*\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*;', re.M)
names = sorted(set(pat.findall(hdr)))

# Collect any undefined identifiers from "using X = Y[IDENT];" declarations
# and emit them as #defines by searching for #define IDENT in the header.
using_array_pat = re.compile(r'^using\s+\w+\s*=\s*\w+\[(\w+)\]\s*;', re.M)
defined_pat = re.compile(r'^#define\s+(\w+)\b', re.M)
defined_names = set(defined_pat.findall(hdr))
undefined_array_dims = [m for m in using_array_pat.findall(hdr) if m not in defined_names]

out = []
out.append('struct BabyBear {')
out.append('  unsigned x;')
out.append('};')
out.append('template <typename T, unsigned N>')
out.append('struct BinomialExtensionField {')
out.append('  T x[N];')
out.append('};')
# Provide fallback definitions for array dimensions that cbindgen
# could not resolve (e.g. constants re-exported from dependencies).
for dim in undefined_array_dims:
    out.append(f'#ifndef {dim}')
    out.append(f'#define {dim} 8')
    out.append(f'#endif')
out.append(f'#include "{hdr_path}"')
out.append('static void force_refs() {')
out.append('  volatile void* sink = 0;')
for n in names:
    out.append(f'  sink = (void*)&{n};')
out.append('  (void)sink;')
out.append('}')
sys.stdout.write("\n".join(out))
