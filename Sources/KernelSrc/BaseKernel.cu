#include "BaseKernel.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


namespace BaseKernel {

	__global__ void mat_add(double* mat1, double* mat2, double* matF, Dimension matSize) {
		int tIdX = blockIdx.x * blockDim.x + threadIdx.x;
		int tIdY = blockIdx.y * blockDim.y + threadIdx.y;
		int tIdZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (tIdX < matSize.width && tIdY < matSize.height && tIdZ < matSize.depth) {
			int linLoc = tIdX + tIdY * matSize.width + tIdZ * matSize.width * matSize.height;
			matF[linLoc] = mat1[linLoc] + mat2[linLoc];
		}
	}

	__global__ void mat_zero(double* mat, Dimension matSize) {
		int tIdX = blockIdx.x * blockDim.x + threadIdx.x;
		int tIdY = blockIdx.y * blockDim.y + threadIdx.y;
		int tIdZ = blockIdx.z * blockDim.z + threadIdx.z;

		if (tIdX < matSize.width && tIdY < matSize.height && tIdZ < matSize.depth) {
			int linLoc = tIdX + tIdY * matSize.width + tIdZ * matSize.width * matSize.height;

			mat[linLoc] = 0;
		}
	}

	__global__ void assign_input(NeuralLayer* d_network, double* input, Dimension input_size) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		if (tIdx < input_size.width && tIdy < input_size.height && tIdz < input_size.depth) {
			int linLoc = tIdx + tIdy * input_size.width + tIdz * input_size.width * input_size.height;

			//printf("Input %f\n", input[linLoc]);

			d_network[0].set_input_value(tIdx, tIdy, tIdz, input[linLoc]);

			//printf("Neuron Input %f\n", d_network[0].get_input_value(tIdx, tIdy, tIdz));
		}
	}

	__global__ void assign_output(NeuralLayer* d_network, double* output, int layers, Dimension output_size) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		if (tIdx < output_size.width && tIdy < output_size.height && tIdz < output_size.depth) {
			int linLoc = tIdx + tIdy * output_size.width + tIdz * output_size.width * output_size.height;
			output[linLoc] = d_network[layers - 1].get_output_value(tIdx, tIdy, tIdz);
		}
	}

	__global__ void apply(NeuralLayer* d_network, int layers) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;

		if (tIdx < layers) {
			dim3 tpb = { 8,8,8 };
			dim3 blocks = { d_network[tIdx].get_dim().width / (tpb.x) + 1, d_network[tIdx].get_dim().height / (tpb.y) + 1, d_network[tIdx].get_dim().depth / (tpb.z) + 1 };
			
			Dimension dim = d_network[tIdx].get_dim();

			
			if (d_network[tIdx].ls.convolutional) {
				d_network[tIdx].convolutional_apply();
			}

			apply_layer << <blocks, tpb >> > (d_network, tIdx );			

			cudaDeviceSynchronize();

			d_network[tIdx].reset();

		}
	}

	__global__ void apply_layer(NeuralLayer* d_network, int layer_id) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;
	
		Dimension dim = d_network[layer_id].get_dim();

		if (tIdx < dim.width && tIdy < dim.height && tIdz < dim.depth) {
			double bias = d_network[layer_id].get_bias_value(tIdx, tIdy, tIdy);
			
			double bias_delta = 0;
			
			if(d_network[layer_id].get_bias_updates(tIdx, tIdy, tIdz) != 0)
				bias_delta = d_network[layer_id].get_bias_delta(tIdx, tIdy, tIdz) / d_network[layer_id].get_bias_updates(tIdx, tIdy, tIdz);

			//printf("Bias Delta %f, Location (%i, {%i, %i, %i})\n", bias_delta, layer_id, tIdx, tIdy, tIdz);

			d_network[layer_id].set_bias_value(tIdx, tIdy, tIdz, bias + bias_delta);

			Dimension obj_dim = d_network[layer_id].get_obj_in_dim();

			dim3 tpb = { 8,8,8 };
			dim3 blocks = { obj_dim.width / tpb.x + 1, obj_dim.height / tpb.y + 1, obj_dim.depth / tpb.z + 1 };

			apply_weight << <blocks, tpb >> > (d_network, layer_id, obj_dim, Vector({ tIdx, tIdy, tIdz }));

			cudaDeviceSynchronize();

			d_network[layer_id].get_weights_in_of(tIdx, tIdy, tIdz).reset();

		}
		

	}

	__global__ void apply_weight(NeuralLayer* d_network, int layer_id, Dimension obj_dim, Vector neuron_id) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		if (tIdx < obj_dim.width && tIdy < obj_dim.height && tIdz < obj_dim.depth) {
			NeuralWeightMatrix to_edit = d_network[layer_id].get_weights_in_of(neuron_id);

			double weight = to_edit.get_weight(tIdx, tIdy, tIdz).weight;

			double weight_delta = 0;

			if(to_edit.get_weight_updates(tIdx, tIdy, tIdz) != 0)
				weight_delta = to_edit.get_weight_delta(tIdx, tIdy, tIdz)/to_edit.get_weight_updates(tIdx, tIdy, tIdz);

			//printf("Weight Delta %f, Location (%i, {%i, %i, %i})\n", weight_delta, layer_id, neuron_id.x, neuron_id.y, neuron_id.z);

			to_edit.set_prev_weight(weight, tIdx, tIdy, tIdz);
			to_edit.set_weight(weight + weight_delta, tIdx, tIdy, tIdz);
		}
	}
};