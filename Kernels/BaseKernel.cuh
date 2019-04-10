#ifndef BASE_KERNEL_CUH
#define BASE_KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorStructs.cuh"
#include "NeuralLayer.cuh"

#include <stdio.h>

#include "glcube.cuh"

namespace BaseKernel {

	__global__ void mat_add(double* mat1, double* mat2, double* matF, Dimension matSize);
	__global__ void mat_zero(double* mat, Dimension matSize);

	__global__ void assign_input(NeuralLayer* d_network, double* input, Dimension input_size);
	__global__ void assign_output(NeuralLayer* d_network, double* output, int layers, Dimension output_size);

	__global__ void apply(NeuralLayer* d_network, int layer);
	__global__ void apply_layer(NeuralLayer* d_network, int layer_id);
	__global__ void apply_weight(NeuralLayer* d_network, int layer_id, Dimension obj_dim, Vector neuron_id);
};
#endif