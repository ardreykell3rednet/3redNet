#ifndef NEURAL_WEIGHT_MATRIX_H
#define NEURAL_WEIGHT_MATRIX_H

#include "VectorStructs.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

class NeuralWeightMatrix {
	public:

		Weight* weights;

		double* weight_deltas;

		double* output_matrix;

		LayerLocation to;

		__device__ __host__ NeuralWeightMatrix();
		__host__ NeuralWeightMatrix(Dimension d, LayerLocation to);
		__host__ NeuralWeightMatrix(unsigned int width, unsigned int height, unsigned int depth, LayerLocation to);

		__host__ void reinitialize();
		__host__ void reinitialize(Connection* connections);
		__host__ void reinitialize(std::vector<Connection> connections);

		__device__ __host__ bool is_initialized();
		
		__device__ __host__ Weight* get_weights();
		__device__ __host__ Weight get_weight(Vector loc);
		__device__ __host__ Weight get_weight(int x, int y, int z);

		__device__ __host__ double get_prev_weight(Vector loc);
		__device__ __host__ double get_prev_weight(int x, int y, int z);

		__device__ __host__ double* get_weight_deltas();
		__device__ __host__ double get_weight_delta(Vector loc);
		__device__ __host__ double get_weight_delta(int x, int y, int z);

		__device__ __host__ double* get_delta_outputs();
		__device__ __host__ double get_delta_output(Vector loc);
		__device__ __host__ double get_delta_output(int x, int y, int z);

		__device__ __host__ void set_weight(double weight, Vector loc);
		__device__ __host__ void set_weight(double weight, int x, int y, int z);

		__device__ __host__ void set_prev_weight(double weight, Vector loc);
		__device__ __host__ void set_prev_weight(double weight, int x, int y, int z);

		__device__ __host__ void set_connection(Connection conn, Vector loc);
		__device__ __host__ void set_connection(Connection conn, int x, int y, int z);

		__device__ __host__ void copy_weights(Weight* weights);

		__device__ __host__ void add_weight_delta(double delta, Vector loc);
		__device__ __host__ void add_weight_delta(double delta, int x, int y, int z);
		__device__ __host__ void set_weight_deltas(double* deltas);

		__device__ __host__ void set_output_at(double output, Vector loc);
		__device__ __host__ void set_output_at(double output, int x, int y, int z);

		__device__ __host__ int get_weight_updates(Vector loc);
		__device__ __host__ int get_weight_updates(int x, int y, int z);

		__device__ __host__ void add_weight_update(Vector loc);
		__device__ __host__ void add_weight_update(int x, int y, int z);
		__device__ __host__ void reset();

		__device__ __host__ LayerLocation get_output_location();

		__device__ __host__ Dimension get_dimension();
		__device__ __host__ void set_dimension(Dimension d);
		__device__ __host__ bool oob_error(Vector location);

	private:

		Dimension dim;


		bool is_init = false;

};

#endif
