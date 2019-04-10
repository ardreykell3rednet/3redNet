#ifndef BACKWARD_PROPAGATION_KERNEL_CUH
#define BACKWARD_PROPAGATION_KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorStructs.cuh"
#include <stdio.h>

#include "NeuralLayer.cuh"
#include "Error.cuh"

namespace BackwardPropagationKernel {

	template <unsigned int block_size>
	__device__ double sum_output(double* input, int size);

	template <unsigned int block_size>
	__global__ void reduce(double* input, double* output, int size);

	//__global__ void hidden_search_parent(NeuralLayer* network, int layer_id, int num_layers, LayerLocation neuron_id, double* final_dirs);
	//__global__ void hidden_search_sum(NeuralLayer* network, int layer_id, Dimension layer_dim, LayerLocation neuron_id, double* err_dirs);

	//__global__ void search_weight_matrix(NeuralLayer* network, LayerLocation search_from, LayerLocation to_search, double* err_dir);

	//__global__ void connection_based_backward_propagation(NeuralLayer * network, NeuralWeightMatrix** connections, Error* net_err, ErrorType net_err_type, double * preferredValues, double computed_err, int num_layers);
	
	//__global__ void connection_based_conjugate_gradient_backpropagation(NeuralLayer* network, NeuralWeightMatrix* connections, Error* net_err, ErrorType net_err_type, double* preferredValues, double computed_err, int num_neurons, int num_layers, int layerId, int load, int block_size);
	//__global__ void update_conjugate_gradient_matrix(NeuralLayer* network, int layer_id, Dimension obj_dim, NeuralWeightMatrix weight_mat, NeuralWeightMatrix conn_mat, double gradient);
	//__global__ void backward_neuron_conjugate_gradient(NeuralLayer* network, int layer_id, Dimension layer_dim, Dimension obj_dim, Vector neuron_id, double comp_err);

	__global__ void connection_based_layer_backpropagation(NeuralLayer* network, NeuralWeightMatrix* connections, Error* net_err, ErrorType net_err_type, double* preferredValues, double computed_err, int num_neurons, int num_layers, int layerId, int load, int block_size);
	__global__ void update_gradient_matrix(NeuralLayer* network, int layer_id, Dimension obj_dim, NeuralWeightMatrix weight_mat, NeuralWeightMatrix conn_mat, double gradient);
	__global__ void backward_neuron_gradient(NeuralLayer* network, int layer_id, Dimension layer_dim, Dimension obj_dim, Vector neuron_id, double comp_err);
};

#endif
	