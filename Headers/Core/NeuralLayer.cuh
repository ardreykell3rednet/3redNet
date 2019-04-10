#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "VectorStructs.cuh"
#include "LayerSettings.cuh"
#include "NeuralWeightMatrix.cuh"
#include "NeuralGradientMatrix.cuh"
#include "Activation.cuh"

#include <stdio.h>

class NeuralLayer {

	public:

		Neuron* neurons;

		NeuralWeightMatrix* weight_in_ptrs;

		NeuralGradientMatrix* gradient_ptrs;

		double* bias_deltas;

		LayerSettings ls;

		__device__ __host__ NeuralLayer();
		__host__ NeuralLayer(Dimension size, int layer_id, LayerSettings ls);
		__host__ NeuralLayer(unsigned int width, unsigned int height, unsigned int depth, int layer_id, LayerSettings ls);

		__host__ void initialize(LayerSettings ls);
		__device__ __host__ void reinitialize();

		__host__ void update_weight_in_matrix_size(Dimension d);

		__device__ __host__ double* get_bias_deltas();
		__device__ __host__ double get_bias_delta(Vector loc);
		__device__ __host__ double get_bias_delta(int x, int y, int z);

		__device__ __host__ int get_bias_updates(Vector loc);
		__device__ __host__ int get_bias_updates(int x, int y, int z);

		__device__ __host__ void add_bias_update(Vector loc);
		__device__ __host__ void add_bias_update(int x, int y, int z);
		__device__ __host__ void reset();

		__device__ __host__ double get_bias_value(Vector loc);
		__device__ __host__ double get_bias_value(int x, int y, int z);

		__device__ __host__ double get_input_value(Vector loc);
		__device__ __host__ double get_input_value(int x, int y, int z);

		__device__ __host__ double get_output_value(Vector loc);
		__device__ __host__ double get_output_value(int x, int y, int z);

		__device__ __host__ double get_gradient_value(Vector loc);
		__device__ __host__ double get_gradient_value(int x, int y, int z);

		__device__ __host__ void add_bias_delta(Vector loc, double delta);
		__device__ __host__ void add_bias_delta(int x, int y, int z, double delta);

		__device__ __host__ void set_input_value(Vector loc, double input);
		__device__ __host__ void set_input_value(int x, int y, int z, double input);

		__device__ __host__ void set_output_value(Vector loc, double output);
		__device__ __host__ void set_output_value(int x, int y, int z, double output);

		__device__ __host__ void set_gradient_value(Vector loc, double gradient);
		__device__ __host__ void set_gradient_value(int x, int y, int z, double gradient);

		__device__ __host__ void set_prev_gradient_value(Vector loc, double gradient);
		__device__ __host__ void set_prev_gradient_value(int x, int y, int z, double gradient);

		__device__ __host__ void set_bias_value(Vector loc, double bias);
		__device__ __host__ void set_bias_value(int x, int y, int z, double bias);

		__device__ __host__ void set_activated(Vector loc, bool activated);
		__device__ __host__ void set_activated(int x, int y, int z, bool activated);

		__device__ __host__ NeuralWeightMatrix* get_weight_in_ptrs();
		
		__device__ __host__ NeuralWeightMatrix get_weights_in_of(Vector loc);
		__device__ __host__ NeuralWeightMatrix get_weights_in_of(int x, int y, int z);

		__device__ __host__ NeuralGradientMatrix* get_gradient_ptrs();

		__device__ __host__ NeuralGradientMatrix get_gradients_of(Vector loc);
		__device__ __host__ NeuralGradientMatrix get_gradients_of(int x, int y, int z);

		__device__ __host__ Neuron get_neuron_at(Vector loc);
		__device__ __host__ Neuron get_neuron_at(int x, int y, int z);

		__device__ __host__ Dimension get_dim();
		__device__ __host__ Dimension get_obj_in_dim();
		__device__ __host__ Dimension get_obj_grad_dim();

		__device__ __host__ Activation get_layer_func();
		__device__ __host__ ActivationType get_layer_func_type();

		__device__  void convolutional_apply();

		__device__ __host__ int get_id();
		__device__ __host__ bool oob_error(Vector location);

	private:
	
		Dimension size;
		Dimension obj_in_dim;
		Dimension obj_grad_dim;

		Activation layer_func;
		ActivationType layer_func_type;

		int id;

		bool init = false;

};


#endif


