#ifndef RECTIFIED_LINEAR_UNITS_H
#define RECTIFIED_LINEAR_UNITS_H

#include "Activation.cuh"

class RectifiedLinearUnits : public Activation {
public:
	__device__ __host__ RectifiedLinearUnits();

	__device__ __host__ float compute(float input);
	__device__ __host__ float derive(float input);
};

#endif
