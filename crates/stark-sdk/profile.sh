#!/bin/bash

# This script profiles an example binary for STARK proving
eg_name=$1

git_root=$(git rev-parse --show-toplevel)
cd $git_root/crates/stark-sdk

arch=$(uname -m)
case $arch in
    arm64|aarch64)
        export RUSTFLAGS="-Ctarget-cpu=native -g -C force-frame-pointers=yes"
        ;;
    x86_64|amd64)
        export RUSTFLAGS="-Ctarget-cpu=native -C target-feature=+avx512f -g -C force-frame-pointers=yes"
        ;;
    *)
        echo "Unsupported architecture: $arch"
        exit 1
        ;;
esac

export JEMALLOC_SYS_WITH_MALLOC_CONF="retain:true,background_thread:true,metadata_thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1,abort_conf:true"

cargo build --profile=profiling --example $eg_name --no-default-features --features=nightly-features,jemalloc,parallel

# Check if samply is installed
if ! command -v samply &> /dev/null; then
    echo "samply not found. Installing..."
    cargo install samply
else
    echo "samply is already installed"
fi


if command -v perf &> /dev/null && [[ "$(uname -s)" == "Linux" ]]; then
    perf record -F 100 --call-graph=fp -g -o perf.data -- $git_root/target/profiling/examples/$eg_name
    samply import perf.data
else
    samply record $git_root/target/profiling/examples/$eg_name
fi
