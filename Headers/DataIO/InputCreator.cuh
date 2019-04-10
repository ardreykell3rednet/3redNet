#ifndef INPUT_CREATOR_CUH
#define INPUT_CREATOR_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Activation.cuh"
#include "VectorStructs.cuh"

#include <string>

namespace InputCreator {
	__device__ __host__ double* create(std::string* parsed_input, int size, Dimension to_fill);
	__device__ __host__ double* create_image(std::string parsed_input, Dimension to_fill);
};

#endif