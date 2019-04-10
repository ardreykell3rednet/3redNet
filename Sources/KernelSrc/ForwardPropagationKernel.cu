#include "ForwardPropagationKernel.cuh"

namespace ForwardPropagationKernel {

	template <unsigned int block_size>
	__device__ double sum_output(double* input, int size) {

		if (size <= 0) {
			return 0;
		}

		if (1 >= size) {

			double finAns = input[0];

			return finAns;
		}
		else {
			int n_size = size / block_size + 1;

			double* n_output = (double*)malloc(n_size * sizeof(double));

			reduce<block_size> << <n_size, block_size >> > (input, n_output, size);

			cudaDeviceSynchronize();

			double finAns = sum_output<block_size>(n_output, n_size);

			free(n_output);

			return finAns;

		}

	}

	template <unsigned int block_size>
	__device__ double max_output(double* input, int size) {

		if (size <= 0) {
			return 0;
		}

		if (1 >= size) {

			double finAns = input[0];

			return finAns;
		}
		else {
			int n_size = size / block_size + 1;

			double* n_output = (double*)malloc(n_size * sizeof(double));

			reduce_max<block_size> << <n_size, block_size >> > (input, n_output, size);

			cudaDeviceSynchronize();

			double finAns = sum_output<block_size>(n_output, n_size);

			free(n_output);

			return finAns;

		}

	}

	template <unsigned int block_size>
	__global__ void reduce(double* input, double* output, int size) {
		__shared__ double sdata[block_size];

		unsigned int tid = threadIdx.x;
		unsigned int index = blockIdx.x * blockDim.x + tid;

		if (index < size) {
			sdata[tid] = input[index];
		}
		else {
			sdata[tid] = 0;
		}

		__syncthreads();

		if (block_size >= 512 && tid < 256) { sdata[tid] += sdata[tid + 256]; __syncthreads(); }
		if (block_size >= 256 && tid < 128) { sdata[tid] += sdata[tid + 128]; __syncthreads(); }
		if (block_size >= 128 && tid < 64) { sdata[tid] += sdata[tid + 64]; __syncthreads(); }

		if (tid < 32) {
			if (block_size >= 64) sdata[tid] += sdata[tid + 32];
			if (block_size >= 32) sdata[tid] += sdata[tid + 16];
			if (block_size >= 16) sdata[tid] += sdata[tid + 8];
			if (block_size >= 8) sdata[tid] += sdata[tid + 4];
			if (block_size >= 4) sdata[tid] += sdata[tid + 2];
			if (block_size >= 2) sdata[tid] += sdata[tid + 1];
		}


		__syncthreads();

		if (tid == 0) output[blockIdx.x] = sdata[0];
	}


	template <unsigned int block_size>
	__global__ void reduce_max(double* input, double* output, int size) {
		__shared__ double sdata[block_size];

		unsigned int tid = threadIdx.x;
		unsigned int index = blockIdx.x * blockDim.x + tid;

		if (index < size) {
			sdata[tid] = input[index];
		}
		else {
			sdata[tid] = 0;
		}

		/*__syncthreads();

		for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
			if (tid < s) {

				if (abs(sdata[tid]) > abs(sdata[tid + s])) {
									
				}
				else {
					sdata[tid] = sdata[tid + s];
				}

				//sdata[tid] = sdata[tid + s] > sdata[tid] ? sdata[tid + s] : sdata[tid];
			}
			__syncthreads();
		}


		__syncthreads();*/

		if (block_size >= 512 && tid < 256) { sdata[tid] = abs(sdata[tid]) > abs(sdata[tid + 256]) ? sdata[tid] : sdata[tid + 256]; __syncthreads(); }
		if (block_size >= 256 && tid < 128) { sdata[tid] = abs(sdata[tid]) > abs(sdata[tid + 128]) ? sdata[tid] : sdata[tid + 128]; __syncthreads(); }
		if (block_size >= 128 && tid < 64) { sdata[tid] = abs(sdata[tid]) > abs(sdata[tid + 64]) ? sdata[tid] : sdata[tid + 64]; __syncthreads(); }

		if (tid < 32) {
			if (block_size >= 64) sdata[tid] = abs(sdata[tid]) > abs(sdata[tid + 32]) ? sdata[tid] : sdata[tid + 32];
			if (block_size >= 32) sdata[tid] = abs(sdata[tid]) > abs(sdata[tid + 16]) ? sdata[tid] : sdata[tid + 16];
			if (block_size >= 16) sdata[tid] = abs(sdata[tid]) > abs(sdata[tid + 8]) ? sdata[tid] : sdata[tid + 8];
			if (block_size >= 8) sdata[tid] = abs(sdata[tid]) > abs(sdata[tid + 4]) ? sdata[tid] : sdata[tid + 4];
			if (block_size >= 4) sdata[tid] = abs(sdata[tid]) > abs(sdata[tid + 2]) ? sdata[tid] : sdata[tid + 2];
			if (block_size >= 2) sdata[tid] = abs(sdata[tid]) > abs(sdata[tid + 1]) ? sdata[tid] : sdata[tid + 1];
		}


		__syncthreads();

		if (tid == 0) output[blockIdx.x] = sdata[0];
	}

	__global__ void forward_layer_propagate(NeuralLayer* network, int layer_id, Dimension layer_dim) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		if (tIdx < layer_dim.width && tIdy < layer_dim.height && tIdz < layer_dim.depth) {

			Vector loc = { tIdx, tIdy, tIdz };

			Dimension w_mat = network[layer_id].get_obj_in_dim();
			
			dim3 tpb = { 8, 8, 8 };
			dim3 blocks = { w_mat.width / tpb.x + 1, w_mat.height / tpb.y + 1, w_mat.depth / tpb.z + 1 };
			double* final_sums = (double*)malloc(blocks.x * blocks.y * blocks.z * sizeof(double));

			forward_neuron_propagate << <blocks, tpb >> > (network, layer_id, layer_dim, w_mat, loc, final_sums);

			cudaDeviceSynchronize();
			
			
			double fin_sum = 0;

			for (int i = 0; i < blocks.x * blocks.y * blocks.z; i++) {
				fin_sum += final_sums[i];
			}
			
			//printf("Fin Sum %f\n", fin_sum);

			if (layer_id != 0) {
				network[layer_id].set_input_value(loc, fin_sum);

				double output = network[layer_id].get_layer_func().compute(fin_sum + network[layer_id].get_bias_value(loc), network[layer_id].get_layer_func_type());

				network[layer_id].set_output_value(loc, output);
			}
			else {
				double output = network[layer_id].get_input_value(loc);

				network[layer_id].set_output_value(loc, output); 

				//printf("Neuron Output %f\n", network[layer_id].get_output_value(tIdx, tIdy, tIdz));
			}
			

			free(final_sums);
		}


	}

	__global__ void forward_neuron_propagate(NeuralLayer* network, int layer_id, Dimension layer_dim, Dimension obj_dim, Vector neuron_id, double* final_sums) {
		__shared__ double block_dat[8 * 8 * 8];

		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;


		if (tIdx < obj_dim.width && tIdy < obj_dim.height && tIdz < obj_dim.depth) {

			NeuralWeightMatrix to_apply = network[layer_id].get_weights_in_of(neuron_id);

			Weight use = to_apply.get_weight(tIdx, tIdy, tIdz);

			//printf("Weight %f\nOutput %f\n", use.weight, network[use.conn.from.layerId].get_output_value(use.conn.from.location));

			LayerLocation from = use.conn.from;

			double output = network[from.layerId].get_output_value(from.location);

			int linLoc = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
			block_dat[linLoc] = use.weight * output;
		}

		__syncthreads();

		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
			double sum = 0;
			for (int i = 0; i < blockDim.x * blockDim.y * blockDim.z; i++) {
				if (i < obj_dim.width * obj_dim.height * obj_dim.depth) {
					sum += block_dat[i];
				}
			}

			//printf("Sum %f\n", sum);

			final_sums[blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y] = sum;
		}

	}
	

	__global__ void connection_based_forward_propagate(NeuralLayer* network, int num_layers) {

		int tId = blockIdx.x * blockDim.x + threadIdx.x;
	
		if (tId < num_layers) {

			Dimension layer_dim = network[tId].get_dim();

			int num_connections = (int) layer_dim.width * layer_dim.height * layer_dim.depth;

			unsigned int block_size = 128;
			unsigned int blocks = num_connections / block_size + 1;

			connection_based_layer_propagate<64> << <blocks, block_size >> > (network, network[tId].weight_in_ptrs, num_connections, tId, 4);

			
		}
	}

	template<unsigned int block_size>
	__global__ void connection_based_layer_propagate(NeuralLayer* network, NeuralWeightMatrix* connections, int num_connections, int id, int load) {

		int tId = blockIdx.x * blockDim.x + threadIdx.x;

		int i = 0;

		while (i < load) {
			if (tId < num_connections) {

				Dimension calc = connections[tId].get_dimension();
				LayerLocation to = connections[tId].to;
				LayerSettings ls = network[tId].ls;

				NeuralWeightMatrix curr = connections[tId];

				//printf("ID: %i\n", to.layerId);

				dim3 tpb = { 4,4,4 };
				dim3 w_mat = { calc.width / tpb.x + 1, calc.height / tpb.y + 1, calc.depth / tpb.z + 1 };

				if (!ls.convolutional) {
					weight_matrix_det_active << <tpb, w_mat >> > (network, curr, curr, calc);
				}
				else {
					NeuralWeightMatrix zero = network[to.layerId].get_weights_in_of(0, 0, to.location.z);
					weight_matrix_det_active << <tpb, w_mat >> > (network, zero, curr, calc);
				}

				Activation layer_func = network[id].get_layer_func();
				ActivationType lft = network[id].get_layer_func_type();

				double* output_matrix = curr.output_matrix;

				double input = 0;

				if (id > 0) {

					double output;

					if (!ls.pool) {
						input = sum_output<128>(output_matrix, calc.width * calc.depth * calc.height);
						output = layer_func.compute(input + network[id].get_bias_value(to.location), lft);

						network[id].set_input_value(to.location, input);
						network[id].set_output_value(to.location, output);
					}
					else {
						input = max_output<128>(output_matrix, calc.width * calc.depth * calc.height);
						output = layer_func.compute(input + network[id].get_bias_value(to.location), lft);

						network[id].set_input_value(to.location, input);
						network[id].set_output_value(to.location, input);
					}
				}
				else if (id == 0) {
					input = network[id].get_input_value(to.location);
					network[id].set_output_value(to.location, input);
				}


			}

			tId += block_size;
			i++;

			if (tId > num_connections) {
				break;
			}


		}
	}

	
	__global__ void weight_matrix_det_active(NeuralLayer* network, NeuralWeightMatrix weight_mat, NeuralWeightMatrix conn_mat, Dimension obj_dim) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;


		if (tIdx < obj_dim.width && tIdy < obj_dim.height && tIdz < obj_dim.depth) {
			LayerLocation from = conn_mat.get_weight(tIdx, tIdy, tIdz).conn.from;

			if (!network[from.layerId].oob_error(from.location)) {
				double output = network[from.layerId].get_output_value(from.location) * weight_mat.get_weight(tIdx, tIdy, tIdz).weight;
				conn_mat.set_output_at(output, tIdx, tIdy, tIdz);
			}
			else {
				conn_mat.set_output_at(0, tIdx, tIdy, tIdz);
			}

		}

		__syncthreads();
	}

};