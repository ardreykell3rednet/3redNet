#include "BackwardPropagationKernel.cuh"


#define BIAS_RATE 0.00005
#define WEIGHT_RATE 0.00005

namespace BackwardPropagationKernel {

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

	/*__global__ void backward_layer_gradient(NeuralLayer* network, Error* net_err, ErrorType net_err_type, double* preferredValues, int layer_id, int num_layers, double computed_err) {

		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		if (tIdx < network[layer_id].get_dim().width && tIdy < network[layer_id].get_dim().height && tIdz < network[layer_id].get_dim().depth) {

			Dimension layer_dim = network[layer_id].get_dim();

			//printf("BACKPROPAGATING %i\n", layer_id);

			if (layer_id == num_layers - 1) {

				int linLoc = tIdx + tIdy * layer_dim.width + tIdz * layer_dim.width * layer_dim.height;

				double err_dir = net_err[0].derive(network[layer_id].get_output_value(tIdx, tIdy, tIdz), preferredValues[linLoc], net_err_type);
				double out_dir = -1 * network[layer_id].get_layer_func().derive(network[layer_id].get_input_value(tIdx, tIdy, tIdz), network[layer_id].get_layer_func_type());

				//printf("ErrDir %f\nOutdir %f\n", err_dir, out_dir);

				network[layer_id].set_gradient_value(tIdx, tIdy, tIdz, out_dir * err_dir);

				network[layer_id].add_bias_delta(tIdx, tIdy, tIdz, network[layer_id].get_gradient_value(tIdx, tIdy, tIdz) * BIAS_RATE * computed_err);
				network[layer_id].add_bias_update(tIdx, tIdy, tIdz);

				Dimension w_mat = network[layer_id].get_obj_in_dim();

				dim3 tpb = { 8, 8, 8 };
				dim3 blocks = { w_mat.width / tpb.x + 1, w_mat.height / tpb.y + 1, w_mat.depth / tpb.z + 1 };

				backward_neuron_gradient << <blocks, tpb >> > (network, layer_id, num_layers, layer_dim, w_mat, Vector({ tIdx, tIdy, tIdz }), computed_err);

			}
			else if(layer_id >= 0){

				double out_dir = -1 * network[layer_id].get_layer_func().derive(network[layer_id].get_input_value(tIdx, tIdy, tIdz), network[layer_id].get_layer_func_type());

				Dimension w_mat = network[layer_id].get_obj_out_dim();

				const dim3 tpb = { 8,8,8 };
				const dim3 layerBlocks = { w_mat.width / tpb.x + 1, w_mat.height / tpb.y + 1, w_mat.depth / tpb.z + 1 };

				double* final_dirs = (double*)malloc(sizeof(double) * layerBlocks.x * layerBlocks.y * layerBlocks.z);

				hidden_matrix_sum << <layerBlocks, tpb >> > (network, network[layer_id].get_neuron_at(tIdx, tIdy, tIdz).location, final_dirs);

				cudaDeviceSynchronize();

				double err_dir = 0;

				for (int i = 0; i < (layerBlocks.x * layerBlocks.y * layerBlocks.z); i++) {
					err_dir += final_dirs[i];
				}

				//printf("ErrDir %f\nOutdir %f\n", err_dir, out_dir);

				network[layer_id].set_gradient_value(tIdx, tIdy, tIdz, err_dir * out_dir);

				network[layer_id].add_bias_delta(tIdx, tIdy, tIdz, err_dir * out_dir * BIAS_RATE);
				network[layer_id].add_bias_update(tIdx, tIdy, tIdz);

				w_mat = network[layer_id].get_obj_in_dim();

				dim3 blocks = { w_mat.width / tpb.x + 1, w_mat.height / tpb.y + 1, w_mat.depth / tpb.z + 1 };

				backward_neuron_gradient << <blocks, tpb >> > (network, layer_id, num_layers, layer_dim, w_mat, Vector({ tIdx, tIdy, tIdz }), computed_err);

				free(final_dirs);

			}

		}

	}

	__global__ void backward_neuron_gradient(NeuralLayer* network, int layer_id, Dimension layer_dim, Dimension obj_dim, Vector neuron_id, double comp_err) {

		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		if (tIdx < obj_dim.width && tIdy < obj_dim.height && tIdz < obj_dim.depth) {

			int linLoc = neuron_id.x + neuron_id.y * layer_dim.width + neuron_id.z * layer_dim.width * layer_dim.height;

			double gradient = network[layer_id].get_gradient_value(neuron_id);

			LayerLocation from = network[layer_id].get_weights_in_of(neuron_id).get_weight(tIdx, tIdy, tIdz).conn.from;

			double from_output = network[from.layerId].get_output_value(from.location);

			network[layer_id].get_weights_in_of(neuron_id).add_weight_delta(gradient * WEIGHT_RATE * from_output * comp_err, tIdx, tIdy, tIdz);
			network[layer_id].get_weights_in_of(neuron_id).add_weight_update(tIdx, tIdy, tIdz);
		}
	}


	__global__ void hidden_matrix_sum(NeuralLayer* network, LayerLocation neuron_id, double* final_dirs) {
		__shared__ double block_dat[512];

		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		Dimension obj_dim = network[neuron_id.layerId].get_obj_out_dim();

		if (tIdx < obj_dim.width && tIdy < obj_dim.height && tIdz < obj_dim.depth) {

			NeuralWeightMatrix mat = network[neuron_id.layerId].get_weights_out_of(neuron_id.location);
			Weight check = mat.get_weight(tIdx, tIdy, tIdz);

			LayerLocation grad_loc = check.conn.to;

			double gradient = network[grad_loc.layerId].get_gradient_value(grad_loc.location);

			int linLoc = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

			double* weight = (double*)malloc(sizeof(double));

			Dimension search_dim = network[grad_loc.layerId].get_obj_in_dim();

			dim3 tpb = { 8,8,8 };
			dim3 blocks = { search_dim.width / tpb.x + 1, search_dim.height / tpb.y + 1, search_dim.depth / tpb.z + 1 };

			search_weight_matrix << <blocks, tpb >> > (network, grad_loc, neuron_id, weight);

			block_dat[linLoc] = gradient * weight[0];

			free(weight);

		}

		__syncthreads();

		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

			double sum = 0;

			for (int i = 0; i < blockDim.x * blockDim.y * blockDim.z; i++) {
				if (i < obj_dim.width * obj_dim.height * obj_dim.depth) {
					sum += block_dat[i];
				}
			}

			final_dirs[blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y] = sum;

		}


	}

	__global__ void search_weight_matrix(NeuralLayer* network, LayerLocation to_search, LayerLocation to_look_for, double* ret) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		Dimension dim = network[to_search.layerId].get_obj_in_dim();

		if (tIdx < dim.width && tIdy < dim.height && tIdz < dim.depth) {

			int linLoc = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

			NeuralWeightMatrix search = network[to_search.layerId].get_weights_in_of(to_search.location);

			LayerLocation other = search.get_weight(tIdx, tIdy, tIdz).conn.from;

			bool is_equal = other.location.x == to_search.location.x && other.location.y == to_search.location.y &&
				other.location.z == to_search.location.z && other.layerId == to_search.layerId;

			if (is_equal) {
				ret[0] = search.get_weight(tIdx, tIdy, tIdz).weight;
				return;
			}
			else {
				return;
			}

		}
	}*/

	__global__ void connection_based_layer_backpropagation(NeuralLayer* network, NeuralWeightMatrix* connections, Error* net_err, ErrorType net_err_type, 
		double* preferredValues, double computed_err, int num_neurons, int num_layers, int layerId, int load, int block_size) {
		
		int tId = blockDim.x * blockIdx.x + threadIdx.x;
		int i = 0;
		while (i < load) {
			if (tId < num_neurons) {

				LayerLocation grad_calc = connections[tId].to;
				LayerSettings ls = network[layerId].ls;

				NeuralWeightMatrix curr = connections[tId];

				double prev_gradient = network[layerId].get_gradient_value(grad_calc.location);

				if (layerId == num_layers - 1) {

					Dimension net_zero = network[layerId].get_dim();

					int lin_loc = grad_calc.location.x + grad_calc.location.y * net_zero.width + grad_calc.location.z * net_zero.width * net_zero.height;

					double output = network[layerId].get_output_value(grad_calc.location);
					double input = network[layerId].get_input_value(grad_calc.location);

					Activation act = network[layerId].get_layer_func();
					ActivationType act_type = network[layerId].get_layer_func_type();

					double dnet_err = net_err[0].derive(output, preferredValues[lin_loc], ErrorType::MSE);
					double dout_err = act.derive(input, act_type);
					double gradient = dnet_err * dout_err;

					network[layerId].set_prev_gradient_value(grad_calc.location, prev_gradient);
					network[layerId].set_gradient_value(grad_calc.location, gradient);
					network[layerId].add_bias_delta(grad_calc.location, -1 * BIAS_RATE * gradient);
					network[layerId].add_bias_update(grad_calc.location);

					Dimension w_mat = network[layerId].get_obj_in_dim();
					dim3 tpb = { 4, 4, 4 };
					dim3 blocks = { w_mat.width / tpb.x + 1, w_mat.height / tpb.y + 1, w_mat.depth / tpb.z + 1 };

					if (!ls.convolutional) {
						update_gradient_matrix << <blocks, tpb >> > (network, layerId, w_mat, curr, curr, gradient);

						cudaDeviceSynchronize();

						backward_neuron_gradient << <blocks, tpb >> > (network, layerId, net_zero, w_mat, grad_calc.location, computed_err);
					}
					else {

						NeuralWeightMatrix zero = network[layerId].get_weights_in_of(0, 0, grad_calc.location.z);

						update_gradient_matrix << <blocks, tpb >> > (network, layerId, w_mat, curr, zero, gradient);

						cudaDeviceSynchronize();

						backward_neuron_gradient << <blocks, tpb >> > (network, layerId, net_zero, w_mat, grad_calc.location, computed_err);
					}
				}
				else {

					Dimension net_id = network[layerId].get_dim();

					double output = network[layerId].get_output_value(grad_calc.location);
					double input = network[layerId].get_input_value(grad_calc.location);

					Activation act = network[layerId].get_layer_func();
					ActivationType act_type = network[layerId].get_layer_func_type();

					Dimension grad_mat_size = network[layerId].get_gradients_of(grad_calc.location).get_dimension();

					NeuralGradientMatrix gradients = network[layerId].get_gradients_of(grad_calc.location);

					double dnet_err = sum_output<128>(gradients.gradient_matrix, grad_mat_size.depth * grad_mat_size.width * grad_mat_size.height);
					double dout_err = act.derive(input, act_type);

					double gradient = dnet_err * dout_err;

					network[layerId].set_prev_gradient_value(grad_calc.location, prev_gradient);
					network[layerId].set_gradient_value(grad_calc.location, gradient);
					network[layerId].add_bias_delta(grad_calc.location, -1 * BIAS_RATE * gradient);
					network[layerId].add_bias_update(grad_calc.location);

					Dimension w_mat = network[layerId].get_obj_in_dim();
					dim3 tpb = { 4, 4, 4 };
					dim3 blocks = { w_mat.width / tpb.x + 1, w_mat.height / tpb.y + 1, w_mat.depth / tpb.z + 1 };

					if (!ls.convolutional) {
						update_gradient_matrix << <blocks, tpb >> > (network, layerId, w_mat, curr, curr, gradient);

						cudaDeviceSynchronize();

						backward_neuron_gradient << <blocks, tpb >> > (network, layerId, net_id, w_mat, grad_calc.location, computed_err);
					}
					else {
						NeuralWeightMatrix zero = network[layerId].get_weights_in_of(0, 0, grad_calc.location.z);

						update_gradient_matrix << <blocks, tpb >> > (network, layerId, w_mat, curr, zero, gradient);

						cudaDeviceSynchronize();

						backward_neuron_gradient << <blocks, tpb >> > (network, layerId, net_id, w_mat, grad_calc.location, computed_err);
					}
				}
			}
		
			i++;
			tId += block_size;


			if (tId > num_neurons) {
				break;
			}
		
		}
	}

	__global__ void update_gradient_matrix(NeuralLayer* network, int layer_id, Dimension obj_dim, NeuralWeightMatrix weight_mat, NeuralWeightMatrix conn_mat, double gradient) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		if (tIdx < obj_dim.width && tIdy < obj_dim.height && tIdz < obj_dim.depth) {

			Weight manage = weight_mat.get_weight(tIdx, tIdy, tIdz);

			LayerLocation behind = conn_mat.get_weight(tIdx, tIdy, tIdz).conn.from;

			if (!network[layer_id].oob_error(behind.location)) {

				double weight = manage.weight;

				Dimension behind_dim = network[behind.layerId].get_dim();

				int behind_lin_loc = behind.location.x + behind.location.y * behind_dim.width + behind.location.z * behind_dim.width * behind_dim.height;

				network[behind.layerId].gradient_ptrs[behind_lin_loc].set_next_gradient_value(gradient * weight);
			}

		}
	}

	
	__global__ void backward_neuron_gradient(NeuralLayer* network,  int layer_id, Dimension layer_dim, Dimension obj_dim, Vector neuron_id, double comp_err) {

		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		if (tIdx < obj_dim.width && tIdy < obj_dim.height && tIdz < obj_dim.depth) {

			int linLoc = neuron_id.x + neuron_id.y * layer_dim.width + neuron_id.z * layer_dim.width * layer_dim.height;

			double gradient = network[layer_id].get_gradient_value(neuron_id);

			LayerLocation from = network[layer_id].get_weights_in_of(neuron_id).get_weight(tIdx, tIdy, tIdz).conn.from;

			if (!network[layer_id].oob_error(from.location)) {
				double from_output = network[from.layerId].get_output_value(from.location);

				network[layer_id].weight_in_ptrs[linLoc].add_weight_delta(-1 * gradient * WEIGHT_RATE * from_output, tIdx, tIdy, tIdz);
			
				network[layer_id].weight_in_ptrs[linLoc].add_weight_update(tIdx, tIdy, tIdz);
			}
			else {
				network[layer_id].weight_in_ptrs[linLoc].add_weight_delta(0, tIdx, tIdy, tIdz);
				network[layer_id].weight_in_ptrs[linLoc].add_weight_update(tIdx, tIdy, tIdz);
			}
		}
	}

	/*
	__global__ void hidden_search_parent(NeuralLayer* network, int layer_id, int num_layers, LayerLocation neuron_id, double* final_dirs) {
		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;

		if (tIdx > layer_id && tIdx < num_layers) {

			Dimension net_dim = network[tIdx].get_dim();

			dim3 tpb = { 8,8,8 };
			dim3 blocks = { net_dim.width / tpb.x + 1, net_dim.height / tpb.y + 1, net_dim.depth / tpb.z + 1 };

			double* err_dirs = (double*)malloc(sizeof(double) * blocks.x * tpb.x * blocks.y * tpb.y * blocks.z * tpb.z);

			hidden_search_sum << <blocks, tpb >> > (network, tIdx, net_dim, neuron_id, err_dirs);

			cudaDeviceSynchronize();

			double sum = 0;

			for (int i = 0; i < blocks.x * tpb.x * blocks.y * tpb.y * blocks.z * tpb.z; i++) {
				sum += err_dirs[i];
			}

			final_dirs[tIdx] = sum;

			free(err_dirs);
		}
		else {
			final_dirs[tIdx] = 0;
		}
	}

	__global__ void hidden_search_sum(NeuralLayer* network, int layer_id, Dimension layer_dim, LayerLocation neuron_id, double* err_dirs) {

		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		if (tIdx < layer_dim.width && tIdy < layer_dim.height && tIdz < layer_dim.depth) {

			int linLoc = tIdx + tIdy * layer_dim.width + tIdz * layer_dim.width * layer_dim.height;

			LayerLocation search_from = { {tIdx, tIdy, tIdz}, layer_id };
			Dimension obj_dim = network[layer_id].get_obj_in_dim();

			dim3 tpb = { 8,8,8 };
			dim3 blocks = { obj_dim.width / tpb.x + 1, obj_dim.height / tpb.y + 1, obj_dim.depth / tpb.z + 1 };

			double* err_dir = (double*)malloc(sizeof(double) * blocks.x * blocks.y * blocks.z);

			//printf("Search from %i\n", search_from.layerId);

			search_weight_matrix << <blocks, tpb>> > (network, search_from, neuron_id, err_dir);

			cudaDeviceSynchronize();

			double sum = 0;

			for (int i = 0; i < blocks.x * blocks.y * blocks.z; i++) {
				sum += err_dir[i];
			}

			err_dirs[linLoc] = sum;

			free(err_dir);

		}
		else {

			int linLoc = tIdx + tIdy * layer_dim.width + tIdz * layer_dim.width * layer_dim.height;

			err_dirs[linLoc] = 0;
		}
	}

	__global__ void search_weight_matrix(NeuralLayer* network, LayerLocation search_from, LayerLocation to_search, double* err_dir) {

		__shared__ double block_dat[8 * 8 * 8];

		int tIdx = blockDim.x * blockIdx.x + threadIdx.x;
		int tIdy = blockDim.y * blockIdx.y + threadIdx.y;
		int tIdz = blockDim.z * blockIdx.z + threadIdx.z;

		Dimension dim = network[search_from.layerId].get_obj_dim();

		if (tIdx < dim.width && tIdy < dim.height && tIdz < dim.depth) {

			int linLoc = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

			NeuralWeightMatrix search = network[search_from.layerId].get_weights_of(search_from.location);

			LayerLocation other = search.get_weight(tIdx, tIdy, tIdz).conn.from;

			bool is_equal = other.location.x == to_search.location.x && other.location.y == to_search.location.y &&
				other.location.z == to_search.location.z && other.layerId == to_search.layerId;

			if (is_equal) {

				LayerLocation to = search.get_weight(tIdx, tIdy, tIdz).conn.to;
				
				double gradient = network[to.layerId].get_gradient_value(to.location);

				block_dat[linLoc] = search.get_weight(tIdx, tIdy, tIdz).weight * gradient;
			}
			else {
				block_dat[linLoc] = 0;
			}
			
		}


		__syncthreads();

		if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {

			double sum = 0;

			for (int i = 0; i < blockDim.x * blockDim.y * blockDim.z; i++) {
				if (i < dim.width * dim.height * dim.depth) {
					sum += block_dat[i];
				}
			}

			err_dir[blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y] = sum;
		}

	}
	*/
};