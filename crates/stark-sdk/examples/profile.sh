#!/bin/bash

# This script profiles an example binary for STARK proving
eg_name=$1

git_root=$(git rev-parse --show-toplevel)
cd $git_root/crates/stark-sdk

arch=$(uname -m)
case $arch in
    arm64|aarch64)
        RUSTFLAGS="-Ctarget-cpu=native -g"
        ;;
    x86_64|amd64)
        RUSTFLAGS="-Ctarget-cpu=native -C target-feature=+avx512f -g"
        ;;
    *)
        echo "Unsupported architecture: $arch"
        exit 1
        ;;
esac

export JEMALLOC_SYS_WITH_MALLOC_CONF="retain:true,background_thread:true,metadata_thp:always,dirty_decay_ms:-1,muzzy_decay_ms:-1,abort_conf:true"

RUSTFLAGS="-C force-frame-pointers=yes" cargo build --profile=profiling --example $eg_name --no-default-features --features=nightly-features,jemalloc,parallel

# Check if samply is installed
if ! command -v samply &> /dev/null; then
    echo "samply not found. Installing..."
    cargo install samply
else
    echo "samply is already installed"
fi

# TODO: linux should use perf with framepointer for better results
samply record $git_root/target/profiling/examples/$eg_name
