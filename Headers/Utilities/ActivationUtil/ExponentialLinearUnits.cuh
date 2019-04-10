#ifndef EXPONENTIAL_LINEAR_UNITS_H
#define EXPONENTIAL_LINEAR_UNITS_H

#include "Activation.cuh"

class ExponentialLinearUnits : public Activation {
	public:
		__device__ __host__ ExponentialLinearUnits();

		__device__ __host__ float compute(float input);
		__device__ __host__ float derive(float input);
};

#endif
