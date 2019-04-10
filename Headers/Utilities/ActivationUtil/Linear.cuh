#ifndef LINEAR_H
#define LINEAR_H

#include "Activation.cuh"

class Linear : public Activation {
public:
	__device__ __host__ Linear();

	__device__ __host__ float compute(float input);
	__device__ __host__ float derive(float input);
};

#endif
