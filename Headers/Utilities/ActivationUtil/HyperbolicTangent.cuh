#ifndef HYPERBOLIC_TANGENT_H
#define HYPERBOLIC_TANGENT_H

#include "Activation.cuh"

class HyperbolicTangent : public Activation {
public:
	__device__ __host__ HyperbolicTangent();

	__device__ __host__ float compute(float input);
	__device__ __host__ float derive(float input);
};

#endif
