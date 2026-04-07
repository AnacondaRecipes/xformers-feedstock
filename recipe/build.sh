set -ex

# Branch on cpu_or_cuda (the variant key) rather than cuda_compiler_version.
# Rationale: xformers' setup.py get_extensions() takes the CUDA path whenever
# TORCH_CUDA_ARCH_LIST or FORCE_CUDA is set in the environment, and
# get_cuda_version() unconditionally calls `nvcc -V` on that path. The CPU
# build worker has no nvcc, so any stray CUDA env var crashes the CPU build.
# Keying off ${cpu_or_cuda} guarantees we only touch the CUDA env on the
# cuda variant. (script_env cannot be used to *override* a variant value —
# it only whitelists vars for pass-through.)
echo "cpu_or_cuda=${cpu_or_cuda}  cuda_compiler_version=${cuda_compiler_version}"

if [[ "${cpu_or_cuda}" == "cuda" ]]; then
    # CUDA arch list per aggregate COOKBOOK §"CUDA Arch Lists by Version":
    #   12.8: adds Blackwell SM 10.0 (10.3/12.x not supported by nvcc 12.8).
    # xformers kernels are useful from Volta (SM70) upward, but we keep the
    # broader arch list for consistency with conda-forge's published builds.
    if [[ "${cuda_compiler_version}" == 12.8 ]]; then
        export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0+PTX"
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
