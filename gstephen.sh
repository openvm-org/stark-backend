#!/bin/bash

cd ~/openvm
RUST_MIN_STACK=8388608 cargo test --package openvm-sdk --test integration_test --features cuda -- test_sdk_standard_with_p256 --exact --show-output; cd ~/stark-backend
