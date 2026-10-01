#!/bin/bash
set -ev

export PATH=/opt/mamba/envs/dp-dev/bin:$PATH
SCRIPT_PATH=/personal/deepmd-kit/.devcontainer
export DP_VARIANT=cuda
export CMAKE_PREFIX_PATH=${SCRIPT_PATH}/../libtorch
export TENSORFLOW_ROOT=/personal/deepmd-kit/.venv/lib/python3.12/site-packages/tensorflow
export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.__path__[0])"):$CMAKE_PREFIX_PATH

CUDAToolkit_ROOT=/opt/mamba/envs/dp-dev
export CUDA_TOOLKIT_ROOT_DIR=$CUDAToolkit_ROOT
export CUDAToolkit_ROOT=$CUDAToolkit_ROOT
export CUDA_INCLUDE_DIRS=$CUDAToolkit_ROOT/include
export CUDA_CUDART_LIBRARY=$CUDAToolkit_ROOT/lib

mkdir -p ${SCRIPT_PATH}/../buildcxx2/
cd ${SCRIPT_PATH}/../buildcxx2/
cmake -D DP_USING_C_API=OFF \
	-D ENABLE_TENSORFLOW=ON \
	-D ENABLE_PYTORCH=ON \
	-D ENABLE_PADDLE=OFF \
	-D USE_CUDA_TOOLKIT=ON \
	-D CMAKE_INSTALL_PREFIX=${SCRIPT_PATH}/../dp/ \
	-D CUDA_TOOLKIT_ROOT_DIR=$CUDAToolkit_ROOT \
	-D CUDAToolkit_ROOT=$CUDAToolkit_ROOT \
	-D CUDA_INCLUDE_DIRS=$CUDA_INCLUDE_DIRS \
	-D CUDA_CUDART_LIBRARY=$CUDA_CUDART_LIBRARY \
	-D LAMMPS_VERSION=stable_22Jul2025_update2 \
	-D CMAKE_BUILD_TYPE=Debug \
	-D BUILD_TESTING:BOOL=TRUE \
	-D TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
	${SCRIPT_PATH}/../source
cmake --build . -j4
cmake --install .
