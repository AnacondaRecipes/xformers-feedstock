@echo on
REM Windows counterpart of build.sh: branch on gpu_variant, the variant key.
echo gpu_variant=%gpu_variant%  cuda_compiler_version=%cuda_compiler_version%

REM setup.py passes MSVC-style flags such as /std:c++17 only when DISTUTILS_USE_SDK=1.
REM The vs2022 activation already provides the MSVC environment; upstream CI sets it too.
set DISTUTILS_USE_SDK=1
set BUILD_VERSION=%package_version%

REM Upstream wheel builds empty this sm90-only cutlass header on Windows to avoid an
REM MSVC compiler bug: .github/workflows/wheels_build.yml "HACKFIX for cutlass compiler bug".
type nul > "%SRC_DIR%\third_party\cutlass\include\cutlass\gemm\kernel\sm90_gemm_tma_warpspecialized_pingpong.hpp"
if errorlevel 1 exit 1

if "%gpu_variant%"=="cuda" goto :cuda
if "%gpu_variant%"=="none" goto :build
echo Unsupported gpu_variant %gpu_variant%
exit 1

:cuda
REM nvcc lives in the build env, not the host env.
set "CUDA_HOME=%BUILD_PREFIX%\Library"
set "CUDA_PATH=%BUILD_PREFIX%\Library"
REM pytorch 2.15 win-64 is built for exactly this list for CUDA 12.9 and 13.x
REM - pytorch-feedstock bld.bat - so kernels for any other arch could never load.
set "cuda_major=%cuda_compiler_version:~0,2%"
if "%cuda_major%"=="12" goto :archlist
if "%cuda_major%"=="13" goto :archlist
echo Unsupported CUDA compiler version %cuda_compiler_version%. Edit bld.bat to add target CUDA archs.
exit 1

:archlist
set "TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;9.0;10.0;12.0+PTX"
set FORCE_CUDA=1
REM Host CUDA import libs live in Library\lib on 12.x and Library\lib\x64 on 13.x.
set "LIB=%PREFIX%\Lib\site-packages\torch\lib;%LIBRARY_LIB%;%LIBRARY_PREFIX%\lib\x64;%LIB%"

:build
%PYTHON% -m pip install . -vv --no-deps --no-build-isolation
if errorlevel 1 exit 1
