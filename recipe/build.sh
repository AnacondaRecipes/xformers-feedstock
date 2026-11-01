set -ex

# Branch on gpu_variant (the variant key) rather than cuda_compiler_version.
# Rationale: xformers' setup.py get_extensions() takes the CUDA path whenever
# TORCH_CUDA_ARCH_LIST or FORCE_CUDA is set in the environment, and
# get_cuda_version() unconditionally calls `nvcc -V` on that path. The CPU
# build worker has no nvcc, so any stray CUDA env var crashes the CPU build.
# Keying off ${gpu_variant} guarantees we only touch the CUDA env on the
# cuda variant. (script_env cannot be used to *override* a variant value —
# it only whitelists vars for pass-through.)
echo "gpu_variant=${gpu_variant}  cuda_compiler_version=${cuda_compiler_version}"

if [[ "${gpu_variant}" == "cuda" ]]; then
    # Floor 7.5: pytorch 2.15 itself ships 7.5+ only. Linux adds native 8.9 (and
    # 10.3/12.1 with nvcc 13) SASS on top of pytorch's 7.5;8.0;8.6;9.0;10.0;12.0+PTX;
    # bld.bat uses pytorch's exact win-64 list.
    if [[ "${cuda_compiler_version}" == 12.* ]]; then
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0+PTX"
    elif [[ "${cuda_compiler_version}" == 13.* ]]; then
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX"
    else
        echo "Unsupported CUDA compiler version: ${cuda_compiler_version}. Edit build.sh to add target CUDA archs."
        exit 1
    fi
    # setup.py reads NVCC_FLAGS (TORCH_NVCC_FLAGS is only read by pytorch's own build).
    export NVCC_FLAGS="${NVCC_FLAGS:-} -Xfatbin -compress-all"
    export FORCE_CUDA=1

    # Compile against the host `cutlass` package instead of the copy bundled in the
    # sdist; setup.py hardcodes the third_party/cutlass include paths.
    rm -rf third_party/cutlass/include third_party/cutlass/tools third_party/cutlass/examples
    mkdir -p third_party/cutlass/tools/util third_party/cutlass/examples
    ln -s "${PREFIX}/include" third_party/cutlass/include
    ln -s "${PREFIX}/include" third_party/cutlass/tools/util/include
fi

# avoid "error: 'value' is unavailable: introduced in macOS 10.13"
export CXXFLAGS="${CXXFLAGS} -D_LIBCPP_DISABLE_AVAILABILITY"

export BUILD_VERSION=${package_version}

$PYTHON -m pip install . -vv --no-deps --no-build-isolation
