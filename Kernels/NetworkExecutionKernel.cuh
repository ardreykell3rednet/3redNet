#ifndef NETWORK_EXECUTION_KERNEL_CUH
#define NETWORK_EXECUTION_KERNEL_CUH

#include "VectorStructs.cuh"

#include "NeuralGradientMatrix.cuh"
#include "NeuralLayer.cuh"

#include "ErrorType.cpp"

#include "InputStorage.cuh"

class NetworkExecutionKernel {

public:

	InputStorage input_stack;

	NeuralLayer* d_network;

	double **d_bias_deltas;
	double **d_weight_deltas;
	double **d_output_matrices;
	double **d_gradient_matrices;
	double **d_beta_matrices;

	Neuron **d_neurons;

	NeuralWeightMatrix **d_weight_in_matrices;
	NeuralGradientMatrix **d_gradients;

	Weight **d_weights_in;

	cudaStream_t stream;
	int execution_device;

	__host__ NetworkExecutionKernel();

	__host__ int malloc(NeuralLayer* network, int layers);

	__host__ double * network_exec(NeuralLayer * host_net, ErrorType err_type, int layers, double * preferred);
	__host__ double* network_exec(NeuralLayer * host_net, ErrorType err_type, int layers);
	__host__ void network_apply(int layers);

	__host__ NeuralLayer * halloc(NeuralLayer* dim, int layers);

	__host__ int free(NeuralLayer* network, int layers);

	__host__ void push(double* input, int size);



};

#endif