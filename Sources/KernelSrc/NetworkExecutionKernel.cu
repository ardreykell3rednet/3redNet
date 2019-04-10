#include "NetworkExecutionKernel.cuh"

#include "BaseKernel.cuh"

#include "ForwardPropagationKernel.cuh"
#include "BackwardPropagationKernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>

#include <iostream>

#define LOAD 4


__host__ NetworkExecutionKernel::NetworkExecutionKernel() {
	input_stack = InputStorage();
}

__host__ int NetworkExecutionKernel::malloc(NeuralLayer* network, int layers) {

	//free(network, layers);

	int num_neurons = 0;

	for (int id = 0; id < layers; id++) {
		for (unsigned int j = 0; j < network[id].get_dim().width * network[id].get_dim().height * network[id].get_dim().depth; j++) {
			num_neurons++;
		}
	}

	d_network = new NeuralLayer[layers];

	d_bias_deltas = new double*[layers];
	d_weight_deltas = new double*[num_neurons];
	d_gradient_matrices = new double*[num_neurons];
	d_output_matrices = new double*[num_neurons];
	d_beta_matrices = new double*[num_neurons];

	d_neurons = new Neuron*[layers];
	d_gradients = new NeuralGradientMatrix*[layers];
	d_weight_in_matrices = new NeuralWeightMatrix*[layers];

	d_weights_in = new Weight*[num_neurons];

	if (cudaMalloc(&d_network, sizeof(NeuralLayer) * layers) != cudaSuccess) {
		printf("Network Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaMemcpy(d_network, network, sizeof(NeuralLayer) * layers, cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Network MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	int inside_track = 0;

	for (int id = 0; id < layers; id++) {

		if (cudaMalloc((&d_neurons[id]), sizeof(Neuron) * network[id].get_dim().width * network[id].get_dim().height * network[id].get_dim().depth) != cudaSuccess) {
			printf("Neuron Layer Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
		}

		if (cudaMemcpy(&(d_network[id].neurons), &d_neurons[id], sizeof(Neuron*), cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Neuron Layer Pointer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
		}

		if (cudaMemcpy(d_neurons[id], network[id].neurons, sizeof(Neuron) * network[id].get_dim().width * network[id].get_dim().height * network[id].get_dim().depth, cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Neuron Layer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
		}


		if (cudaMalloc(&(d_bias_deltas[id]), sizeof(double) * network[id].get_dim().width * network[id].get_dim().height * network[id].get_dim().depth) != cudaSuccess) {
			printf("Bias Delta Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
		}

		if (cudaMemcpy(&(d_network[id].bias_deltas), &d_bias_deltas[id], sizeof(double*), cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Neuron Layer Pointer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
		}

		if (cudaMemcpy(d_bias_deltas[id], network[id].bias_deltas, sizeof(double) * network[id].get_dim().width * network[id].get_dim().height * network[id].get_dim().depth, cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Bias Delta MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
		}


		if (cudaMalloc(&(d_weight_in_matrices[id]), sizeof(NeuralWeightMatrix) * network[id].get_dim().width * network[id].get_dim().height * network[id].get_dim().depth) != cudaSuccess) {
			printf("Linker Weight Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
		}

		if (cudaMemcpy(&(d_network[id].weight_in_ptrs), &d_weight_in_matrices[id], sizeof(NeuralWeightMatrix*), cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Neuron Layer Pointer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
		}

		if (cudaMemcpy(d_weight_in_matrices[id], network[id].weight_in_ptrs, sizeof(NeuralWeightMatrix) * network[id].get_dim().width * network[id].get_dim().height * network[id].get_dim().depth, cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Linker Weight MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
		}


		if (cudaMalloc(&(d_gradients[id]), sizeof(NeuralGradientMatrix) * network[id].get_dim().width * network[id].get_dim().height * network[id].get_dim().depth) != cudaSuccess) {
			printf("Linker Weight Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
		}

		if (cudaMemcpy(&(d_network[id].gradient_ptrs), &d_gradients[id], sizeof(NeuralGradientMatrix*), cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Neuron Layer Pointer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
		}

		if (cudaMemcpy(d_gradients[id], network[id].gradient_ptrs, sizeof(NeuralGradientMatrix) * network[id].get_dim().width * network[id].get_dim().height * network[id].get_dim().depth, cudaMemcpyHostToDevice) != cudaSuccess) {
			printf("Linker Weight MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
		}


		Dimension net_dim = network[id].get_dim();


		for (unsigned int fin = 0; fin < net_dim.width * net_dim.height * net_dim.depth; fin++) {

			Dimension obj_in_dim = network[id].get_obj_in_dim();
			Dimension grad_dim = network[id].get_obj_grad_dim();

			if (cudaMalloc(&(d_weights_in[inside_track]), sizeof(Weight) * obj_in_dim.width * obj_in_dim.height * obj_in_dim.depth) != cudaSuccess) {
				printf("Weight Matrix Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(&(d_weight_in_matrices[id][fin].weights), &d_weights_in[inside_track], sizeof(Weight*), cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Neuron Layer Pointer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(d_weights_in[inside_track], network[id].weight_in_ptrs[fin].weights, sizeof(Weight) * obj_in_dim.width * obj_in_dim.height * obj_in_dim.depth, cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Weight Matrix MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}


			if (cudaMalloc(&(d_weight_deltas[inside_track]), sizeof(double) * obj_in_dim.width * obj_in_dim.height * obj_in_dim.depth) != cudaSuccess) {
				printf("Weight Delta Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(&(d_weight_in_matrices[id][fin].weight_deltas), &d_weight_deltas[inside_track], sizeof(double*), cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Neuron Layer Pointer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(d_weight_deltas[inside_track], network[id].weight_in_ptrs[fin].weight_deltas, sizeof(double) * obj_in_dim.width * obj_in_dim.height * obj_in_dim.depth, cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Weight Delta MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}


			if (cudaMalloc(&(d_output_matrices[inside_track]), sizeof(double) * obj_in_dim.width * obj_in_dim.height * obj_in_dim.depth) != cudaSuccess) {
				printf("Weight Delta Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(&(d_weight_in_matrices[id][fin].output_matrix), &d_output_matrices[inside_track], sizeof(double*), cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Neuron Layer Pointer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(d_output_matrices[inside_track], network[id].weight_in_ptrs[fin].output_matrix, sizeof(double) * obj_in_dim.width * obj_in_dim.height * obj_in_dim.depth, cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Weight Delta MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}


			if (cudaMalloc(&(d_gradient_matrices[inside_track]), sizeof(double) * grad_dim.width * grad_dim.height * grad_dim.depth) != cudaSuccess) {
				printf("Weight Delta Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(&(d_gradients[id][fin].gradient_matrix), &d_gradient_matrices[inside_track], sizeof(double*), cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Neuron Layer Pointer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(d_gradient_matrices[inside_track], network[id].gradient_ptrs[fin].gradient_matrix, sizeof(double) * grad_dim.width * grad_dim.height * grad_dim.depth, cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Weight Delta MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMalloc(&(d_beta_matrices[inside_track]), sizeof(double) * grad_dim.width * grad_dim.height * grad_dim.depth) != cudaSuccess) {
				printf("Weight Delta Malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(&(d_gradients[id][fin].beta_matrix), &d_beta_matrices[inside_track], sizeof(double*), cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Neuron Layer Pointer MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			if (cudaMemcpy(d_beta_matrices[inside_track], network[id].gradient_ptrs[fin].beta_matrix, sizeof(double) * grad_dim.width * grad_dim.height * grad_dim.depth, cudaMemcpyHostToDevice) != cudaSuccess) {
				printf("Weight Delta MCPY: %s\n", cudaGetErrorString(cudaGetLastError()));
			}

			inside_track++;
		}

	}


	return 0;
}

__host__ double * NetworkExecutionKernel::network_exec(NeuralLayer* host_net, ErrorType err_type, int layers, double* preferred) {

	Dimension tpb = { 4,4,4 };
	Dimension zero_dim = host_net[0].get_dim();

	double* d_input = input_stack.get_next();

	//printf("Input (%f, %f)\n", input[0], input[1]);


	/*if (cudaMalloc(&d_input, sizeof(double) * host_net[0].get_dim().width * host_net[0].get_dim().height * host_net[0].get_dim().depth) != cudaSuccess) {
		printf("Input Malloc %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaMemcpy(d_input, input, sizeof(double) * host_net[0].get_dim().width * host_net[0].get_dim().height * host_net[0].get_dim().depth, cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Input MCPY %s\n", cudaGetErrorString(cudaGetLastError()));
	}*/

	clock_t start = std::clock();

	Dimension n_blocks = (zero_dim / tpb) + Dimension({ 1, 1, 1 });
	dim3 b = { n_blocks.width, n_blocks.height, n_blocks.depth };
	dim3 t = { 4,4,4 };
	cudaSetDevice(execution_device);
	BaseKernel::assign_input << <b, t, 0, stream >> > (d_network, d_input, host_net[0].get_dim());
	cudaDeviceSynchronize();

	float seconds = std::clock() / (float)CLOCKS_PER_SEC - start / (float)CLOCKS_PER_SEC;

	printf("  Forward Prep :> %f  ", seconds);

	start = std::clock();

	for (int i = 0; i < layers; i++) {

		/*Dimension fin = host_net[i].get_dim();
		n_blocks = (fin / tpb) + Dimension({ 1,1,1 });

		dim3 b = { n_blocks.width, n_blocks.height, n_blocks.depth };

		ForwardPropagationKernel::forward_layer_propagate << <b, t >> > (d_network, i, fin);
		cudaDeviceSynchronize();*/

		Dimension size = host_net[i].get_dim();
		int threads = 64;

		int neurons = size.width * size.height * size.depth;

		int blocks = neurons / (threads * LOAD) + 1;
		cudaSetDevice(execution_device);
		ForwardPropagationKernel::connection_based_layer_propagate<64> << <blocks, threads, 0, stream >> > (d_network, d_weight_in_matrices[i], neurons, i, LOAD);

		cudaDeviceSynchronize();

	}

	seconds = std::clock() / (float)CLOCKS_PER_SEC - start / (float)CLOCKS_PER_SEC;

	printf("  Forward Exec :> %f  ", seconds);

	start = std::clock();

	Dimension output_dim = host_net[layers - 1].get_dim();
	double* output = new double[output_dim.width * output_dim.height * output_dim.depth];

	double* d_output;
	//double* outputF = new double[output_dim.width * output_dim.height * output_dim.depth];

	if (cudaMalloc(&d_output, sizeof(double) * output_dim.width * output_dim.height * output_dim.depth) != cudaSuccess) {
		printf("Output Malloc %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	Dimension final_dim = host_net[layers - 1].get_dim();
	n_blocks = (final_dim / tpb) + Dimension({ 1, 1, 1 });
	b = { n_blocks.width, n_blocks.height, n_blocks.depth };
	t = { 4,4,4 };
	cudaSetDevice(execution_device);
	BaseKernel::assign_output << <b, t, 0, stream >> > (d_network, d_output, layers, output_dim);

	if (cudaMemcpy(output, d_output, sizeof(double) * output_dim.width * output_dim.height * output_dim.depth, cudaMemcpyDeviceToHost) != cudaSuccess) {
		printf("Output MCPY %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaFree(d_output) != cudaSuccess) {
		printf("Output Free\n");
	}

	Error net_err = Error();

	double computed_err = 0.0;

	for (int i = 0; i < output_dim.width * output_dim.height * output_dim.depth; i++) {
		computed_err += net_err.compute(output[i], preferred[i], ErrorType::MSE);
	}

	Error* d_net_err;

	double* d_preferred;

	if (cudaMalloc(&d_net_err, sizeof(Error)) != cudaSuccess) {
		printf("Error Malloc %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaMemcpy(d_net_err, &net_err, sizeof(Error), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Error MCPY %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaMalloc(&d_preferred, sizeof(preferred)) != cudaSuccess) {
		printf("Preferred Malloc %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaMemcpy(d_preferred, preferred, sizeof(preferred), cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Preferred MCPY %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	seconds = std::clock() / (float)CLOCKS_PER_SEC - start / (float)CLOCKS_PER_SEC;

	printf("  Backward Prep	 :> %f  ", seconds);

	clock_t start2 = std::clock();
	for (int i = layers - 1; i >= 0; i--) {

		Dimension size = host_net[i].get_dim();
		int threads = 64;

		int neurons = size.width * size.height * size.depth;

		int blocks = neurons / (threads * LOAD) + 1;
		cudaSetDevice(execution_device);
		BackwardPropagationKernel::connection_based_layer_backpropagation << <blocks, threads, 0, stream >> > (d_network, d_weight_in_matrices[i], d_net_err, ErrorType::MSE, d_preferred, computed_err, neurons, layers, i, LOAD, threads);
		cudaDeviceSynchronize();
	}

	float seconds2 = std::clock() / (float)CLOCKS_PER_SEC - start2 / (float)CLOCKS_PER_SEC;

	printf("  Backward Exec :> %f  ", seconds2);

	if (cudaFree(d_net_err) != cudaSuccess) {
		printf("Error Free %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaFree(d_preferred) != cudaSuccess) {
		printf("Preferred Value Free %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	return output;


}

__host__ double* NetworkExecutionKernel::network_exec(NeuralLayer* host_net, ErrorType err_type, int layers) {
	Dimension tpb = { 4,4,4 };
	Dimension zero_dim = host_net[0].get_dim();

	double* d_input = input_stack.get_next();

	//printf("Input (%f, %f)\n", input[0], input[1]);


	/*if (cudaMalloc(&d_input, sizeof(double) * host_net[0].get_dim().width * host_net[0].get_dim().height * host_net[0].get_dim().depth) != cudaSuccess) {
		printf("Input Malloc %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaMemcpy(d_input, input, sizeof(double) * host_net[0].get_dim().width * host_net[0].get_dim().height * host_net[0].get_dim().depth, cudaMemcpyHostToDevice) != cudaSuccess) {
		printf("Input MCPY %s\n", cudaGetErrorString(cudaGetLastError()));
	}*/

	clock_t start = std::clock();

	Dimension n_blocks = (zero_dim / tpb) + Dimension({ 1, 1, 1 });
	dim3 b = { n_blocks.width, n_blocks.height, n_blocks.depth };
	dim3 t = { 4,4,4 };
	BaseKernel::assign_input << <b, t, 0, stream >> > (d_network, d_input, host_net[0].get_dim());
	cudaDeviceSynchronize();

	float seconds = std::clock() / (float)CLOCKS_PER_SEC - start / (float)CLOCKS_PER_SEC;

	printf("  Forward Prep :> %f  ", seconds);

	start = std::clock();

	for (int i = 0; i < layers; i++) {

		/*Dimension fin = host_net[i].get_dim();
		n_blocks = (fin / tpb) + Dimension({ 1,1,1 });

		dim3 b = { n_blocks.width, n_blocks.height, n_blocks.depth };

		ForwardPropagationKernel::forward_layer_propagate << <b, t >> > (d_network, i, fin);
		cudaDeviceSynchronize();*/

		Dimension size = host_net[i].get_dim();
		int threads = 64;

		int neurons = size.width * size.height * size.depth;

		int blocks = neurons / (threads * LOAD) + 1;

		ForwardPropagationKernel::connection_based_layer_propagate<64> << <blocks, threads, 0, stream >> > (d_network, d_weight_in_matrices[i], neurons, i, LOAD);

		cudaDeviceSynchronize();

	}

	seconds = std::clock() / (float)CLOCKS_PER_SEC - start / (float)CLOCKS_PER_SEC;

	printf("  Forward Exec :> %f  ", seconds);

	start = std::clock();

	Dimension output_dim = host_net[layers - 1].get_dim();
	double* output = new double[output_dim.width * output_dim.height * output_dim.depth];

	double* d_output;

	if (cudaMalloc(&d_output, sizeof(double) * output_dim.width * output_dim.height * output_dim.depth) != cudaSuccess) {
		printf("Output Malloc %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	Dimension final_dim = host_net[layers - 1].get_dim();
	n_blocks = (final_dim / tpb) + Dimension({ 1, 1, 1 });
	b = { n_blocks.width, n_blocks.height, n_blocks.depth };
	t = { 4,4,4 };
	BaseKernel::assign_output << <b, t, 0, stream >> > (d_network, d_output, layers, output_dim);

	if (cudaMemcpy(output, d_output, sizeof(double) * output_dim.width * output_dim.height * output_dim.depth, cudaMemcpyDeviceToHost) != cudaSuccess) {
		printf("Output MCPY %s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (cudaFree(d_output) != cudaSuccess) {
		printf("Output Free\n");
	}

	return output;
}

__host__ void NetworkExecutionKernel::network_apply(int layers) {

	const int tpb = 10;
	const int blocks = layers / tpb + 1;

	cudaSetDevice(execution_device);
	BaseKernel::apply << <blocks, tpb, 0, stream >> > (d_network, layers);
}

__host__ NeuralLayer* NetworkExecutionKernel::halloc(NeuralLayer* dim, int layers) {
	NeuralLayer* network = new NeuralLayer[layers];

	int inside_track = 0;

	for (int i = 0; i < layers; i++) {

		Dimension layer_dim = dim[i].get_dim();

		if (cudaMemcpy((network + i), (d_network + i), sizeof(NeuralLayer), cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Network Halloc Error\n"); }

		network[i].neurons = new Neuron[layer_dim.height * layer_dim.width * layer_dim.depth];
		network[i].bias_deltas = new double[layer_dim.height * layer_dim.width * layer_dim.depth];
		network[i].weight_in_ptrs = new NeuralWeightMatrix[layer_dim.height * layer_dim.width * layer_dim.depth];
		network[i].gradient_ptrs = new NeuralGradientMatrix[layer_dim.height * layer_dim.width * layer_dim.depth];

		if (cudaMemcpy((network[i].neurons), (d_neurons[i]), sizeof(Neuron) * layer_dim.depth * layer_dim.height * layer_dim.width, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Neuron Halloc Error\n"); }
		if (cudaMemcpy((network[i].bias_deltas), (d_bias_deltas[i]), sizeof(double) * layer_dim.depth * layer_dim.height * layer_dim.width, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Bias Halloc Error\n"); }
		if (cudaMemcpy((network[i].weight_in_ptrs), (d_weight_in_matrices[i]), sizeof(NeuralWeightMatrix) * layer_dim.depth * layer_dim.height * layer_dim.width, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Weight Matrix Halloc Error\n"); }
		if (cudaMemcpy((network[i].gradient_ptrs), (d_gradients[i]), sizeof(NeuralGradientMatrix) * layer_dim.depth * layer_dim.height * layer_dim.width, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Gradient Matrix Halloc Error\n"); }

		for (int inner = 0; inner < layer_dim.width * layer_dim.height * layer_dim.depth; inner++) {

			Dimension obj_dim = dim[i].get_obj_in_dim();
			Dimension grad_dim = dim[i].get_obj_grad_dim();

			double* gradient_matrix = new double[grad_dim.depth * grad_dim.height * grad_dim.width];
			double* beta_matrix = new double[grad_dim.depth * grad_dim.height * grad_dim.width];
			double* output_matrix = new double[obj_dim.width * obj_dim.height * obj_dim.depth];
			Weight* weight_matrix = new Weight[obj_dim.width * obj_dim.height * obj_dim.depth];
			double* weight_deltas = new double[obj_dim.width * obj_dim.height * obj_dim.depth];

			if (cudaMemcpy(gradient_matrix, (d_gradient_matrices[inside_track]), sizeof(double) * grad_dim.depth * grad_dim.height * grad_dim.width, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Gradient Matrix-D Halloc Error\n"); }
			if (cudaMemcpy(beta_matrix, (d_beta_matrices[inside_track]), sizeof(double) * grad_dim.depth * grad_dim.height * grad_dim.width, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Gradient Matrix-D Halloc Error\n"); }
			if (cudaMemcpy(output_matrix, (d_output_matrices[inside_track]), sizeof(double) * obj_dim.depth * obj_dim.height * obj_dim.width, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Output Matrix-D Halloc Error\n"); }
			if (cudaMemcpy(weight_matrix, (d_weights_in[inside_track]), sizeof(Weight) * obj_dim.depth * obj_dim.height * obj_dim.width, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Weight Matrix-D Halloc Error\n"); }
			if (cudaMemcpy(weight_deltas, (d_weight_deltas[inside_track]), sizeof(double) * obj_dim.depth * obj_dim.height * obj_dim.width, cudaMemcpyDeviceToHost) != cudaSuccess) { printf("Weight Delta Matrix-D Halloc Error\n"); }

			network[i].gradient_ptrs[inner].gradient_matrix = gradient_matrix;
			network[i].gradient_ptrs[inner].beta_matrix = beta_matrix;
			network[i].weight_in_ptrs[inner].output_matrix = output_matrix;
			network[i].weight_in_ptrs[inner].weights = weight_matrix;
			network[i].weight_in_ptrs[inner].weight_deltas = weight_deltas;

			inside_track++;

		}

	}

	return network;
}


__host__ int NetworkExecutionKernel::free(NeuralLayer* network, int layers) {

	int inside_track = 0;

	for (int id = 0; id < layers; id++) {


		if (cudaFree(d_neurons[id]) != cudaSuccess) {
			printf("Neuron Free");
		}

		if (cudaFree(d_bias_deltas[id]) != cudaSuccess) {
			printf("Bias Free");
		}

		if (cudaFree(d_weight_in_matrices[id]) != cudaSuccess) {
			printf("Weight_M Free");
		}

		if (cudaFree(d_gradients[id]) != cudaSuccess) {
			printf("Weight_M Free");
		}

		Dimension net_dim = network[id].get_dim();

		for (unsigned int fin = 0; fin < net_dim.width * net_dim.height * net_dim.depth; fin++) {

			if (cudaFree(d_weight_deltas[inside_track]) != cudaSuccess) {
				printf("Weight_D Free");
			}

			if (cudaFree(d_weights_in[inside_track]) != cudaSuccess) {
				printf("Weight Free");
			}

			if (cudaFree(d_output_matrices[inside_track]) != cudaSuccess) {
				printf("Matrix Free");
			}

			if (cudaFree(d_gradient_matrices[inside_track]) != cudaSuccess) {
				printf("Matrix 2 Free");
			}

			inside_track++;
		}
	}

	if (cudaFree(d_network) != cudaSuccess) {
		printf("NETWORK FREE\n");
	}


	return 0;
}

__host__ void NetworkExecutionKernel::push(double* push, int size) {
	input_stack.push_back(push, size);
}

