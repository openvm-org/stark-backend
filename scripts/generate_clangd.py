#! /usr/bin/env python3

import os
import yaml

def collect_include_dirs(base_path):
    include_dirs = []
    for root, dirs, _ in os.walk(base_path):
        include_dirs.append(f"-I{root}")
    return include_dirs

def main():
    """
    Generate a .clangd configuration file for the workspace. Note that this should
    be run from the root of the workspace.
    """
    workspace_root = os.getcwd()

    # Fixed includes (relative to workspace)
    fixed_includes = [
        os.path.join(workspace_root, "crates/cuda-backend/cuda/include"),
        os.path.join(workspace_root, "crates/cuda-backend/cuda/supra/include"),
        os.path.join(workspace_root, "crates/cuda-common/include"),
    ]

    # Recursive includes
    recursive_includes = []

    all_includes = [f"-I{path}" for path in fixed_includes] + recursive_includes

    # Final .clangd dictionary
    clangd_config = {
        "CompileFlags": {
            "Add": all_includes + ["-x", "cuda", "-std=c++17"]
        },
        "Diagnostics": {
            "UnusedIncludes": "Strict"
        }
    }

    with open(".clangd", "w") as f:
        yaml.dump(clangd_config, f, sort_keys=False)

    print("✅ .clangd file generated successfully.")

if __name__ == "__main__":
    main()
