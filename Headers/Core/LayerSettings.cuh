#ifndef LAYER_SETTINGS_H
#define LAYER_SETTINGS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Activation.cuh"
#include "VectorStructs.cuh"


struct LayerSettings {
	bool includeBias;
	bool convolutional;
	bool pool;
	ActivationType activation;
	Dimension weight_in_dim;
	Dimension grad_dim;
};

#endif