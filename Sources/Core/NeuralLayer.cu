#include "NeuralLayer.cuh"
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "BaseKernel.cuh"

#include <stdio.h>

__global__ void applyBias(Neuron* one, double* two, int size) {
	int tId = blockDim.x * blockIdx.x + threadIdx.x;

	if (tId < size) {
		one[tId].bias += two[tId];
	}

}

__device__ __host__ NeuralLayer::NeuralLayer() {
	this->size = { 0,0,0 };
	this->neurons = NULL;
	this->id = 0;
}

__host__ NeuralLayer::NeuralLayer(Dimension size, int layer_id, LayerSettings ls) {
	this->id = layer_id;
	this->size = size;

	this->neurons = new Neuron[size.depth * size.height * size.width];
	this->bias_deltas = new double[size.depth * size.height * size.width];

	this->weight_in_ptrs = new NeuralWeightMatrix[size.depth * size.height * size.width];
	this->gradient_ptrs = new NeuralGradientMatrix[size.depth * size.height * size.width];

	initialize(ls);

}

__host__ NeuralLayer::NeuralLayer(unsigned int width, unsigned int height, unsigned int depth, int layer_id, LayerSettings ls) {
	this->id = layer_id;
	this->size = { width, height, depth };

	this->neurons = new Neuron[size.depth * size.height * size.width];
	this->bias_deltas = new double[size.depth * size.height * size.width];

	this->weight_in_ptrs = new NeuralWeightMatrix[size.depth * size.height * size.width];
	this->gradient_ptrs = new NeuralGradientMatrix[size.depth * size.height * size.width];

	initialize(ls);

}

__host__ void NeuralLayer::initialize(LayerSettings ls) {
	for (unsigned int x = 0; x < size.width; x++) {
		for (unsigned int y = 0; y < size.height; y++) {
			for (unsigned int z = 0; z < size.depth; z++) {

				int i = x + y * size.width + z * size.width * size.height;

				if (ls.includeBias) {
					neurons[i].bias = ((double)(rand() % 100) / 100) - 0.5;
				}
				else {
					neurons[i].bias = 0;
				}

				neurons[i].gradient = 0;
				neurons[i].input = 0;
				neurons[i].output = 0;
				neurons[i].location = { Vector({(int)x, (int)y, (int)z}), id };

				neurons[i].activated = false;
				neurons[i].learned = false;
				neurons[i].grad_applied = false;

				neurons[i].bias_updates = 0;

				neurons[i].id = i;

				bias_deltas[i] = 0;

				gradient_ptrs[i] = NeuralGradientMatrix(ls.grad_dim, neurons[i].location);

				//printf("Neuron %i Created at Location: (%i, {%i, %i, %i})\n", neurons[i].id, neurons[i].location.layerId,
					//neurons[i].location.location.x, neurons[i].location.location.y, neurons[i].location.location.z);

			}
		}
	}

	this->ls = ls;

	layer_func = Activation();
	layer_func_type = ls.activation;

	this->obj_grad_dim = ls.grad_dim;

	update_weight_in_matrix_size(ls.weight_in_dim);

	init = true;

}

__device__ __host__ void NeuralLayer::reinitialize() {
	for (unsigned int i = 0; i < size.width * size.height * size.depth; i++) {
		bias_deltas[i] = 0;
	}
}


__host__ void NeuralLayer::update_weight_in_matrix_size(Dimension d){
	this->obj_in_dim = d;

	init = true;

	for (unsigned int i = 0; i < size.depth * size.width * size.height; i++) {
		weight_in_ptrs[i] = NeuralWeightMatrix(d, neurons[i].location);
	}


}

__device__ __host__ double* NeuralLayer::get_bias_deltas() {
	return bias_deltas;
}

__device__ __host__ double NeuralLayer::get_bias_delta(Vector loc) {
	return  oob_error(loc) ? NULL : bias_deltas[loc.x + loc.y * size.width + loc.z * size.width * size.height];
}

__device__ __host__ double NeuralLayer::get_bias_delta(int x, int y, int z) {
	Vector v{ x,y,z };
	return get_bias_delta(v);
}

__device__ __host__ int NeuralLayer::get_bias_updates(Vector loc){
	return oob_error(loc) ? 0 : neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].bias_updates;
}

__device__ __host__ int NeuralLayer::get_bias_updates(int x, int y, int z){
	Vector v = { x,y,z };
	return get_bias_updates(v);
}

__device__ __host__ void NeuralLayer::add_bias_update(Vector loc){
	neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].bias_updates += 1;
}

__device__ __host__ void NeuralLayer::add_bias_update(int x, int y, int z){
	Vector v = { x,y,z };
	add_bias_update(v);
}

__device__ __host__ void NeuralLayer::reset()
{
	for (unsigned int x = 0; x < size.width; x++) {
		for (unsigned int y = 0; y < size.height; y++) {
			for (unsigned int z = 0; z < size.depth; z++) {

				int i = x + y * size.width + z * size.width * size.height;

				neurons[i].gradient = 0;
				neurons[i].input = 0;
				neurons[i].output = 0;

				neurons[i].activated = false;
				neurons[i].learned = false;
				neurons[i].grad_applied = false;

				neurons[i].bias_updates = 0;

				bias_deltas[i] = 0;

				gradient_ptrs[i].reset();

			}
		}
	}
}

__device__ __host__ double NeuralLayer::get_bias_value(Vector loc) {
	return oob_error(loc) ? NULL : neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].bias;
}

__device__ __host__ double NeuralLayer::get_bias_value(int x, int y, int z) {
	Vector v{ x, y, z };
	return get_bias_value(v);
}

__device__ __host__ double NeuralLayer::get_input_value(Vector loc) {
	return  oob_error(loc) ? NULL : neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].input;
}

__device__ __host__ double NeuralLayer::get_input_value(int x, int y, int z) {
	Vector v{ x,y,z };
	return get_input_value(v);
}

__device__ __host__ double NeuralLayer::get_output_value(Vector loc) {
	return  oob_error(loc) ? NULL : neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].output;
}

__device__ __host__ double NeuralLayer::get_output_value(int x, int y, int z) {
	Vector v{ x,y,z };
	return get_output_value(v);
}

__device__ __host__ double NeuralLayer::get_gradient_value(Vector loc) {
	return oob_error(loc) ? NULL : neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].gradient;
}

__device__ __host__ double NeuralLayer::get_gradient_value(int x, int y, int z) {
	Vector v{ x, y, z };
	return get_gradient_value(v);
}

__device__ __host__ void NeuralLayer::add_bias_delta(Vector loc, double delta) {
	if (!oob_error(loc)) {
		bias_deltas[loc.x + loc.y * size.width + loc.z * size.width * size.height] += delta;
		//add_bias_update(loc);
	}
	else {
		printf("N_LAYER->DIM ERROR\n");
	}
}


__device__ __host__ void NeuralLayer::add_bias_delta(int x, int y, int z, double delta) {
	Vector v{ x,y,z };
	add_bias_delta(v, delta);
}

__device__ __host__ void NeuralLayer::set_input_value(Vector loc, double input) {
	if (!oob_error(loc)) {
		neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].input = input;
	}
	else {
		printf("N_LAYER->DIM ERROR\n");
	}
}


__device__ __host__ void NeuralLayer::set_input_value(int x, int y, int z, double input) {
	Vector v{ x, y, z };
	set_input_value(v, input);
}

__device__ __host__ void NeuralLayer::set_output_value(Vector loc, double output) {
	if (!oob_error(loc)) {
		neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].output = output;
	}
	else {
		printf("N_LAYER->DIM ERROR\n");
	}
}

__device__ __host__ void NeuralLayer::set_output_value(int x, int y, int z, double output) {
	Vector v{ x, y, z };
	set_output_value(v, output);
}

__device__ __host__ void NeuralLayer::set_gradient_value(Vector loc, double gradient) {
	neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].gradient = gradient;
}

__device__ __host__ void NeuralLayer::set_gradient_value(int x, int y, int z, double gradient) {
	Vector v{ x, y, z };
	set_gradient_value(v, gradient);
}

__device__ __host__ void NeuralLayer::set_prev_gradient_value(Vector loc, double gradient) {
	neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].prev_gradient = gradient;
}

__device__ __host__ void NeuralLayer::set_prev_gradient_value(int x, int y, int z, double gradient) {
	Vector v{ x, y, z };
	set_prev_gradient_value(v, gradient);
}

__device__ __host__ void NeuralLayer::set_bias_value(Vector loc, double bias){
	neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].bias = bias;
}

__device__ __host__ void NeuralLayer::set_bias_value(int x, int y, int z, double bias){
	Vector v{ x, y, z };
	set_bias_value(v, bias);
}

__device__ __host__ void NeuralLayer::set_activated(Vector loc, bool activated)
{
	neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height].activated = activated;
}

__device__ __host__ void NeuralLayer::set_activated(int x, int y, int z, bool activated)
{
	Vector v = { x,y,z };
	set_activated(v, activated);
}

__device__ __host__ NeuralWeightMatrix* NeuralLayer::get_weight_in_ptrs(){
	return weight_in_ptrs;
}

__device__ __host__ NeuralWeightMatrix NeuralLayer::get_weights_in_of(Vector loc) {
	return oob_error(loc) ? NeuralWeightMatrix() : weight_in_ptrs[loc.x + loc.y * size.width + loc.z * size.width * size.height];
}

__device__ __host__ NeuralWeightMatrix NeuralLayer::get_weights_in_of(int x, int y, int z) {
	Vector v{ x, y, z };
	return get_weights_in_of(v);
}

__device__ __host__ NeuralGradientMatrix * NeuralLayer::get_gradient_ptrs()
{
	return gradient_ptrs;
}

__device__ __host__ NeuralGradientMatrix NeuralLayer::get_gradients_of(Vector loc)
{
	return oob_error(loc) ? NeuralGradientMatrix() : gradient_ptrs[loc.x + loc.y * size.width + loc.z * size.width * size.height];
}

__device__ __host__ NeuralGradientMatrix NeuralLayer::get_gradients_of(int x, int y, int z)
{
	Vector v = { x,y,z };
	return get_gradients_of(v);
}

__device__ __host__ Neuron NeuralLayer::get_neuron_at(Vector loc) {
	return oob_error(loc) ? Neuron({NULL, 0, 0, 0, 0, true, true, true}) : neurons[loc.x + loc.y * size.width + loc.z * size.width * size.height];
}

__device__ __host__ Neuron NeuralLayer::get_neuron_at(int x, int y, int z) {
	Vector v{ x, y, z };
	return get_neuron_at(v);
}

__device__ __host__ Dimension NeuralLayer::get_dim() {
	return size;
}

__device__ __host__ Dimension NeuralLayer::get_obj_in_dim(){
	return obj_in_dim;
}

__device__ __host__ Dimension NeuralLayer::get_obj_grad_dim(){
	return obj_grad_dim;
}

__device__ __host__ Activation NeuralLayer::get_layer_func() {
	return layer_func;
}

__device__ __host__ ActivationType NeuralLayer::get_layer_func_type(){
	return layer_func_type;
}

__device__  void NeuralLayer::convolutional_apply()
{
	for (int z = 0; z < get_dim().depth; z++) {
		
		double* final_sum = get_weights_in_of(0, 0, z).weight_deltas;

		for (int x = 0; x < get_dim().width; x++) {
			for (int y = 1; y < get_dim().height; y++) {

				dim3 tpb = { 4,4,4 };
				dim3 n_blocks{ obj_in_dim.width / tpb.x + 1,obj_in_dim.height / tpb.y + 1, obj_in_dim.depth / tpb.z + 1 };

				BaseKernel::mat_add << <n_blocks, tpb >> > (final_sum, get_weights_in_of(x,y,z).weight_deltas, final_sum, obj_in_dim);
				
				cudaDeviceSynchronize();
			}
		}

		for (int x = 0; x < get_dim().width; x++) {
			for (int y = 1; y < get_dim().height; y++) {
				get_weights_in_of(x, y, z).set_weight_deltas(final_sum);
			}
		}

	}
}

__device__ __host__ int NeuralLayer::get_id() {
	return id;
}

__device__ __host__ bool NeuralLayer::oob_error(Vector loc) {
	if (loc.x >= size.width || loc.y >= size.height || loc.z >= size.depth) {
		return true;
	}
	else if (loc.x < 0 || loc.y < 0 || loc.z < 0) {
		return true;
	}

	return false;
}







