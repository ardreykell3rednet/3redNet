#include "NeuralWeightMatrix.cuh"
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__device__ __host__ NeuralWeightMatrix::NeuralWeightMatrix() {
	Dimension d = { 0,0,0 };
	
	this->dim = d;
}
__host__ NeuralWeightMatrix::NeuralWeightMatrix(Dimension d, LayerLocation to) {
	this->dim = d;

	this->weights = new Weight[d.width * d.height * d.depth];
	this->weight_deltas = new double[d.width * d.height * d.depth];
	this->output_matrix = new double[d.width * d.height * d.depth];

	this->to = to;

	//reinitialize();
}

__host__ NeuralWeightMatrix::NeuralWeightMatrix(unsigned int width, unsigned int height, unsigned int depth, LayerLocation to) {
	this->dim = { width, height, depth };

	this->weights = new Weight[dim.width * dim.height * dim.depth];
	this->weight_deltas = new double[dim.width * dim.height * dim.depth];
	this->output_matrix = new double[dim.width * dim.height * dim.depth];
	//reinitialize();

	this->to = to;
}

__host__ void NeuralWeightMatrix::reinitialize() {
	for (unsigned int i = 0; i < dim.width * dim.depth * dim.height; i++) {
		weights[i] = { {}, double((rand() % 100) / 100.0 - 0.5), false };
		weight_deltas[i] = 0;
	}
}

__host__ void NeuralWeightMatrix::reinitialize(Connection* connections) {
	for (unsigned int i = 0; i < dim.width * dim.depth * dim.height; i++) {
		weights[i] = { connections[i], double((rand() % 100) / 100.0), false};
		
		//this->to = connections[i].to;

	//	printf("Connections: Neuron (%i, {%i, %i, %i}) to Neuron (%i,  {%i, %i, %i})\n", connections[i].from.layerId, connections[i].from.location.x, connections[i].from.location.y, connections[i].from.location.z, 
		//	connections[i].to.layerId, connections[i].to.location.x, connections[i].to.location.y, connections[i].to.location.z);
		
		weight_deltas[i] = 0;
	}

	is_init = true;
}

__host__ void NeuralWeightMatrix::reinitialize(std::vector<Connection> connections)
{
	for (unsigned int i = 0; i < dim.width * dim.depth * dim.height; i++) {
		weights[i] = { connections[i], double((rand() % 100) / 100.0), false };

		//this->to = connections[i].to;

		//printf("Connections: Neuron %i to Neuron %i\n", connections[i].from.layerId, connections[i].to.layerId);

		weight_deltas[i] = 0;
	}

	is_init = true;
}

__device__ __host__ bool NeuralWeightMatrix::is_initialized() {
	return is_init;
}

__device__ __host__ Weight* NeuralWeightMatrix::get_weights() {
	return weights;
}

__device__ __host__ Weight NeuralWeightMatrix::get_weight(Vector loc) {
	return oob_error(loc) ? Weight({ {}, 0.0, false }) : weights[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height];
}

__device__ __host__ Weight NeuralWeightMatrix::get_weight(int x, int y, int z) {
	Vector v = { x, y, z };
	return get_weight(v);
}

__device__ __host__ double NeuralWeightMatrix::get_prev_weight(Vector loc) {
	return oob_error(loc) ? 0 : weights[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height].prev_weight;
}

__device__ __host__ double NeuralWeightMatrix::get_prev_weight(int x, int y, int z) {
	Vector v = { x, y, z };
	return get_prev_weight(v);
}

__device__ __host__ double* NeuralWeightMatrix::get_weight_deltas() {
	return weight_deltas;
}

__device__ __host__ double NeuralWeightMatrix::get_weight_delta(Vector loc) {
	return oob_error(loc) ? NULL : weight_deltas[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height];
}

__device__ __host__ double NeuralWeightMatrix::get_weight_delta(int x, int y, int z) {
	Vector v = { x, y, z };
	return get_weight_delta(v);
}

__device__ __host__ double * NeuralWeightMatrix::get_delta_outputs()
{
	return output_matrix;
}

__device__ __host__ double NeuralWeightMatrix::get_delta_output(Vector loc)
{
	return oob_error(loc) ? NULL : output_matrix[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height];
}

__device__ __host__ double NeuralWeightMatrix::get_delta_output(int x, int y, int z)
{
	Vector v = { x, y, z };
	return get_delta_output(v);
}

__device__ __host__ void NeuralWeightMatrix::set_weight(double weight, Vector loc) {
	weights[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height].weight = weight;
}

__device__ __host__ void NeuralWeightMatrix::set_weight(double weight, int x, int y, int z) {
	Vector v = { x, y, z };
	set_weight(weight, v);
}

__device__ __host__ void NeuralWeightMatrix::set_prev_weight(double weight, Vector loc) {
	weights[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height].prev_weight = weight;
}

__device__ __host__ void NeuralWeightMatrix::set_prev_weight(double weight, int x, int y, int z) {
	Vector v = { x, y, z };
	set_prev_weight(weight, v);
}


__device__ __host__ void NeuralWeightMatrix::set_connection(Connection conn, Vector loc)
{
	if (!oob_error(loc)) {
		weights[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height].conn = conn;
	}
}

__device__ __host__ void NeuralWeightMatrix::set_connection(Connection conn, int x, int y, int z)
{
	Vector v = { x,y,z };
	set_connection(conn, v);
}

__device__ __host__ void NeuralWeightMatrix::copy_weights(Weight* weights) {
	for (unsigned int i = 0; i < dim.depth * dim.width * dim.height; i++) {
		this->weights[i].weight = weights[i].weight;
	}
}

__device__ __host__ void NeuralWeightMatrix::add_weight_delta(double delta, Vector loc) {
	if (!oob_error(loc)) {
		weight_deltas[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height] += delta;
		//add_weight_update(loc);
	}
	else {
		printf("N:WEIGHT->DIM ERROR\n");
	}
}

__device__ __host__ void NeuralWeightMatrix::add_weight_delta(double delta, int x, int y, int z) {
	Vector v = { x, y, z };
	add_weight_delta(delta, v);
}

__device__ __host__ void NeuralWeightMatrix::set_weight_deltas(double* deltas) {
	this->weight_deltas = deltas;
}

__device__ __host__ void NeuralWeightMatrix::set_output_at(double output, Vector loc)
{
	if (!oob_error(loc)) {
		output_matrix[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height] = output;
	}
}

__device__ __host__ void NeuralWeightMatrix::set_output_at(double output, int x, int y, int z)
{
	Vector v = { x,y,z };
	set_output_at(output, v);
}

__device__ __host__ int NeuralWeightMatrix::get_weight_updates(Vector loc)
{
	return oob_error(loc) ? 0 : weights[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height].weight_updates;
}

__device__ __host__ int NeuralWeightMatrix::get_weight_updates(int x, int y, int z)
{
	Vector v = { x,y,z };
	return get_weight_updates(v);
}

__device__ __host__ void NeuralWeightMatrix::add_weight_update(Vector loc)
{
	weights[loc.x + loc.y * dim.width + loc.z * dim.width * dim.height].weight_updates += 1;
}

__device__ __host__ void NeuralWeightMatrix::add_weight_update(int x, int y, int z)
{
	Vector v = { x,y,z };
	add_weight_update(v);
}

__device__ __host__ void NeuralWeightMatrix::reset()
{
	for (unsigned int i = 0; i < dim.width * dim.depth * dim.height; i++) {
		weights[i].weight_updates = 0;
		weight_deltas[i] = 0;
	}
}

__device__ __host__ LayerLocation NeuralWeightMatrix::get_output_location()
{
	return to;
}

__device__ __host__ Dimension NeuralWeightMatrix::get_dimension() {
	return dim;
}

__device__ __host__ void NeuralWeightMatrix::set_dimension(Dimension d) {
	this->dim = d;
}


__device__ __host__ bool NeuralWeightMatrix::oob_error(Vector loc) {
	if (loc.x >= dim.width || loc.y >= dim.height || loc.z >= dim.depth) {
		return true;
	}
	else if (loc.x < 0 || loc.y < 0 || loc.z < 0) {
		return true;
	}

	return false;
}


