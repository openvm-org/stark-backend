#!/bin/bash

# This script profiles an example binary for STARK proving
eg_name=$1

git_root=$(git rev-parse --show-toplevel)
cd $git_root/crates/stark-sdk

arch=$(uname -m)
case $arch in
    arm64|aarch64)
        export RUSTFLAGS="-Ctarget-cpu=native"
        ;;
    x86_64|amd64)
        export RUSTFLAGS="-Ctarget-cpu=native -C target-feature=+avx512f"
        ;;
    *)
        echo "Unsupported architecture: $arch"
        exit 1
        ;;
esac

export JEMALLOC_SYS_WITH_MALLOC_CONF="retain:true,background_thread:true,metadata_thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1,abort_conf:true"

cargo run --profile=maxperf --example $eg_name --no-default-features --features=jemalloc,parallel
