#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#include "NeuralLayer.cuh"
#include "ConnectionFormat.cpp"
#include "LayerSettings.cuh"
#include "NetworkExecutionKernel.cuh"
#include "ConnectionFactory.cuh"
#include "InputStorage.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorStructs.cuh"

#include <stdio.h>

#include <ctime>

class NeuralNetwork {
public:

	bool update_required = false;

	double* output;

	int layers;

	std::string name;

	NetworkExecutionKernel nem;

	__device__ __host__ NeuralNetwork();
	__device__ __host__ NeuralNetwork(int layers, Dimension* layer_dims, LayerSettings* layer_settings, ConnectionFormat* connections, std::string name);

	__device__ __host__ NeuralLayer get_layer(int layer_index);
	__device__ __host__ void set_layer(int layer_index, NeuralLayer to_set);
	__device__ __host__ void set_network(NeuralLayer* network);

	__device__ __host__ ConnectionFormat get_connection_format(int layer_index);
	__device__ __host__ void set_connection_format(int layer_index, ConnectionFormat cf);
	__device__ __host__ void set_network_connection_properties(ConnectionFormat* cf);

	__device__ __host__ LayerSettings get_layer_settings(int layer_index);
	__device__ __host__ void set_layer_settings(int layer_index, LayerSettings ls);
	__device__ __host__ void set_network_layer_settings(LayerSettings* ls);

	__device__ __host__ Dimension get_dim(int layer_index);
	__device__ __host__ void set_dim(int layer_index, Dimension size);
	__device__ __host__ void set_network_dim(Dimension* sizes);

	__device__ __host__ void set_stream(cudaStream_t &stream);
	__device__ __host__ void set_execution_device(int device);

	__device__ __host__ bool oob_error(int layer_index);

	__device__ __host__ void push_input(double* input);
	__device__ __host__ void execute(double* preferred);
	__device__ __host__ void execute();
	__device__ __host__ void apply();

	__device__ __host__ void prepare_network();
	__device__ __host__ void connect();
	__device__ __host__ void malloc();

	__device__ __host__ void save();
	//__device__ __host__ void load_from_file();

private:
	NeuralLayer* network;
	ConnectionFormat* connections;
	LayerSettings* layer_settings;
	Dimension* layer_dimensions;

	Dimension input_dimension;

	cudaStream_t stream;

	int execution_device;

};

#endif