#ifndef FORWARD_PROPAGATION_KERNEL_CUH
#define FORWARD_PROPAGATION_KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorStructs.cuh"
#include <stdio.h>

#include "NeuralLayer.cuh"


namespace ForwardPropagationKernel {

	template <unsigned int block_size>
	__device__ double sum_output(double* input, int size);

	template <unsigned int block_size>
	__global__ void reduce(double* input, double* output, int size);

	template <unsigned int block_size>
	__device__ double max_output(double* input, int size);

	template <unsigned int block_size>
	__global__ void reduce_max(double* input, double* output, int size);

	__global__ void forward_layer_propagate(NeuralLayer* network, int layer_id, Dimension layer_dim);
	__global__ void forward_neuron_propagate(NeuralLayer* network, int layer_id, Dimension layer_dim, Dimension obj_dim, Vector neuron_id, double* final_sums);

	__global__ void connection_based_forward_propagate(NeuralLayer* network, int num_layers);
	
	template<unsigned int block_size>
	__global__ void connection_based_layer_propagate(NeuralLayer* network, NeuralWeightMatrix* connections, int num_connections, int id, int load);
	__global__ void weight_matrix_det_active(NeuralLayer* network, NeuralWeightMatrix to_calc, NeuralWeightMatrix conn_mat, Dimension obj_dim);

};

#endif;