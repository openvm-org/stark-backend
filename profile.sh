mkdir -p profiler-runs/nsys-frac-sumcheck

VPMM_PAGE_SIZE=$((4 << 20)) VPMM_PAGES=$((16 << 8)) RUST_LOG=warn \
  nsys profile \
  --output profiler-runs/nsys-frac-sumcheck/full-frac0.01 \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --gpu-metrics-devices=cuda-visible \
  --gpu-metrics-frequency=20000 \
  --cuda-memory-usage=true \
  target/release/synthetic_runner \
  --profile benchmarks/synthetic/reth-block-23992138-profile.jsonl \
  --sample-frac 0.01 --seed 42 --max-log-height 22 \
  --out profiler-runs/nsys-frac-sumcheck/scorecard-frac0.1.json
