#ifndef SIGMOID_H
#define SIGMOID_H

#include "Activation.cuh"

class Sigmoid : public Activation {
public:
	__device__ __host__ Sigmoid();

	__device__ __host__ float compute(float input);
	__device__ __host__ float derive(float input);
};

#endif
