#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "ActivationType.cpp"


class Activation {

	public:
		__device__ __host__ Activation();

		__device__ __host__ double compute(double input, ActivationType at);
		__device__ __host__ double derive(double input, ActivationType at);
};

#endif