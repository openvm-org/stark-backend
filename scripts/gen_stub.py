import re, sys, pathlib

hdr_path = sys.argv[1]
hdr = pathlib.Path(hdr_path).read_text()

# lenient: find lines like "... name(args);", avoiding macros/braces
pat = re.compile(r'^[^#\n{};]*\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*;', re.M)
names = sorted(set(pat.findall(hdr)))

out = []
out.append('struct BabyBear {')
out.append('  unsigned x;')
out.append('};')
out.append('template <typename T, unsigned N>')
out.append('struct BinomialExtensionField {')
out.append('  T x[N];')
out.append('};')
out.append(f'#include "{hdr_path}"')
out.append('static void force_refs() {')
out.append('  volatile void* sink = 0;')
for n in names:
    out.append(f'  sink = (void*)&{n};')
out.append('  (void)sink;')
out.append('}')
sys.stdout.write("\n".join(out))
