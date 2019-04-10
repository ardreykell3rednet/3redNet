#ifndef NEURAL_GRADIENT_MATRIX_CUH
#define NEURAL_GRADIENT_MATRIX_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorStructs.cuh"

#include <stdio.h>

class NeuralGradientMatrix {
	
	public:

		double* gradient_matrix;
		double* beta_matrix;

		LayerLocation to;

		__device__ __host__ NeuralGradientMatrix();

		__device__ __host__ NeuralGradientMatrix(Dimension d, LayerLocation to_relay);

		__device__ __host__ NeuralGradientMatrix(unsigned int width, unsigned int height, unsigned int depth, LayerLocation to_relay);


		__device__ __host__ double* get_gradient_matrix();

		__device__ __host__ double get_gradient_value(Vector loc);
		__device__ __host__ double get_gradient_value(int x, int y, int z);

		__device__ __host__ void set_gradient_value(Vector loc, double value);
		__device__ __host__ void set_next_gradient_value(double value);
		__device__ __host__ void set_gradient_value(int x, int y, int z, double value);

		__device__ __host__ double* get_beta_matrix();

		__device__ __host__ double get_beta_value(Vector loc);
		__device__ __host__ double get_beta_value(int x, int y, int z);

		__device__ __host__ void set_beta_value(Vector loc, double value);
		__device__ __host__ void set_next_beta_value(double value);
		__device__ __host__ void set_beta_value(int x, int y, int z, double value);
		
		__device__ __host__ LayerLocation get_application_location();
		__device__ __host__ Dimension get_dimension();

		__device__ __host__ void set_dimension(Dimension d);

		__device__ __host__ bool is_full();

		__device__ __host__ void reset();

		__device__ __host__ bool oob_error(Vector loc);

	private:

		Dimension grad_dim;

		int num_accesses = 0;
		
};



#endif
