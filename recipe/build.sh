set -ex

# Branch on gpu_variant, not cuda_compiler_version: setup.py takes the CUDA path (and
# runs `nvcc -V`) whenever FORCE_CUDA or TORCH_CUDA_ARCH_LIST is set, and CPU builds
# have no nvcc.
echo "gpu_variant=${gpu_variant}  cuda_compiler_version=${cuda_compiler_version}"

if [[ "${gpu_variant}" == "cuda" ]]; then
    # Start from pytorch 2.15's own lists (pytorch-feedstock build.sh), with two changes:
    # - 9.0a, not 9.0: cutlass compiles the sm90 sparse24 FP8 GEMM only for
    #   arch-specific sm_90a, and leaves a stub that aborts under plain sm_90
    #   (upstream wheels_build.yml also builds 9.0a).
    # - x86_64 also gets native 8.9 SASS, plus 10.3 and 12.1 with nvcc 13.
    if [[ "${target_platform}" == "linux-aarch64" && "${cuda_compiler_version}" == 13.* ]]; then
        export TORCH_CUDA_ARCH_LIST="8.0;9.0a;10.0;11.0;12.0;12.1+PTX"
    elif [[ "${target_platform}" == "linux-64" && "${cuda_compiler_version}" == 12.* ]]; then
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0a;10.0;12.0+PTX"
    elif [[ "${target_platform}" == "linux-64" && "${cuda_compiler_version}" == 13.* ]]; then
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0a;10.0;10.3;12.0;12.1+PTX"
    else
        echo "No CUDA arch list for ${target_platform} CUDA ${cuda_compiler_version}. Edit build.sh to add one."
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
