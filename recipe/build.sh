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
    # CUDA arch list per aggregate COOKBOOK §"CUDA Arch Lists by Version":
    #   12.x: Maxwell→Blackwell SM 10.0 (10.3/12.x not supported by nvcc 12.8/12.9).
    #   13.0: drops Maxwell/Pascal/Volta (5.x/6.x/7.0); min Turing (7.5).
    if [[ "${cuda_compiler_version}" == 12.* ]]; then
        export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0+PTX"
    elif [[ "${cuda_compiler_version}" == 13.* ]]; then
        export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX"
    else
        echo "Unsupported CUDA compiler version: ${cuda_compiler_version}. Edit build.sh to add target CUDA archs."
        exit 1
    fi
    export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
    export FORCE_CUDA=1
fi

# avoid "error: 'value' is unavailable: introduced in macOS 10.13"
export CXXFLAGS="${CXXFLAGS} -D_LIBCPP_DISABLE_AVAILABILITY"

export BUILD_VERSION=${package_version}

$PYTHON -m pip install . -vv --no-deps --no-build-isolation
