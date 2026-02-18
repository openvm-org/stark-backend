"""
Generate a C++ stub that forces the linker to retain every extern "C" function
declared in the cbindgen-generated header, so abi_cmp.py can later read their
DWARF signatures.

Usage:  python3 gen_stub.py <rust_abi.hpp>  >  rust_stub.cpp

The output is compiled with:
    g++ -DCB_TEST -O0 -g -c -fno-eliminate-unused-debug-types rust_stub.cpp

== How the Rust/CUDA ABI matcher pipeline works ==

  1. cbindgen  (configured by scripts/cbindgen.toml)
     Parses the cuda-backend crate **and** its dependencies listed in
     [parse].include (currently openvm-stark-sdk, openvm-stark-backend).
     Produces rust_abi.hpp with C++ type aliases and extern "C" declarations.

  2. gen_stub.py  (this file)
     Wraps rust_abi.hpp in a small .cpp that provides placeholder struct
     definitions for opaque Rust types (BabyBear, BinomialExtensionField, …)
     and a force_refs() function that takes the address of every extern symbol
     so g++ emits DWARF info for them.

  3. g++  – compiles the stub into rust_stub.o (with full debug info).

  4. cargo build (CUDA_DEBUG=1) – produces .o files for the CUDA kernels.

  5. abi_cmp.py  – reads DWARF from both object files and compares function
     signatures (argument count + widths).

== Maintenance notes ==

* Field type structs (BabyBear, BinomialExtensionField)
  Must match the in-memory layout of the Rust types.  BabyBear is a single
  u32 in Montgomery form; BinomialExtensionField<T, N> is T[N].

* Type aliases (F, EF, Digest) — the "fallback types" block
  cbindgen sees the type aliases (F, EF, Digest) re-exported in
  cuda-backend/src/types.rs.  Because the SDK defines these aliases in
  *multiple* config modules (baby_bear_poseidon2 AND baby_bear_bn254_poseidon2,
  the latter behind #[cfg(feature)]), cbindgen deduplicates them and gates
  ALL copies behind the feature's preprocessor guard.  Since the CI compiles
  with only -DCB_TEST (not -DBABY_BEAR_BN254_POSEIDON2), the types vanish.

  The fallback block below provides the default BabyBear-Poseidon2 values so
  the stub always compiles.  These must stay in sync with the constants in
  crates/stark-sdk/src/config/baby_bear_poseidon2.rs:
    - F          = BabyBear
    - EF         = BinomialExtensionField<BabyBear, 4>   (D_EF = 4)
    - DIGEST_SIZE = 8  (== CHUNK)
    - Digest     = F[DIGEST_SIZE]

  If you add a new SDK config (like baby_bear_bn254_poseidon2) and cbindgen.toml
  maps its feature to a C preprocessor symbol, you may need to add a
  corresponding #elif / #else block here for that config's Digest definition.

* Adding a new feature flag
  1. Add the feature → C-define mapping in scripts/cbindgen.toml  [defines].
  2. If the new config changes F, EF, or Digest, add a fallback branch here.
  3. Run the pipeline locally to verify:
       cd crates/cuda-backend
       cbindgen -c ../../scripts/cbindgen.toml . -o rust_abi.hpp
       python3 ../../scripts/gen_stub.py rust_abi.hpp > rust_stub.cpp
       g++ -DCB_TEST -O0 -g -c -fno-eliminate-unused-debug-types rust_stub.cpp

* Array dimension fallbacks
  cbindgen sometimes emits `using X = T[IDENT];` where IDENT is a constant it
  could not resolve (e.g. re-exported from a dependency).  We auto-detect these
  and emit `#define IDENT 8` as a safe default.  The exact value only matters
  for the aggregate byte-width seen by abi_cmp.py — if a mismatch appears,
  check whether the constant needs an explicit #define with the real value.

Related files:
  - scripts/cbindgen.toml         — cbindgen configuration & feature→define map
  - scripts/abi_cmp.py            — DWARF-based ABI comparator
  - crates/cuda-backend/src/types.rs — canonical type aliases used by FFI fns
  - crates/stark-sdk/src/config/  — SDK config modules that define F/EF/Digest
  - .github/workflows/rust-cuda-matcher.yml — CI workflow that runs this pipeline
"""
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

# -- Opaque Rust type placeholders (must match in-memory layout) -----------
out.append('struct BabyBear {')
out.append('  unsigned x;')
out.append('};')
out.append('template <typename T, unsigned N>')
out.append('struct BinomialExtensionField {')
out.append('  T x[N];')
out.append('};')

# -- Fallback type aliases for the default (BabyBear-Poseidon2) config -----
# See "Type aliases" in the docstring above for why this is needed.
out.append('#ifndef BABY_BEAR_BN254_POSEIDON2')
out.append('using F = BabyBear;')
out.append('using EF = BinomialExtensionField<BabyBear, 4>;')
out.append('#define DIGEST_SIZE 8')
out.append('using Digest = F[DIGEST_SIZE];')
out.append('#endif')

# -- Fallback array dimension constants ------------------------------------
for dim in undefined_array_dims:
    out.append(f'#ifndef {dim}')
    out.append(f'#define {dim} 8')
    out.append(f'#endif')

out.append(f'#include "{hdr_path}"')

# -- force_refs: take the address of every extern fn so DWARF is emitted ---
out.append('static void force_refs() {')
out.append('  volatile void* sink = 0;')
for n in names:
    out.append(f'  sink = (void*)&{n};')
out.append('  (void)sink;')
out.append('}')
sys.stdout.write("\n".join(out))
