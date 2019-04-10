#ifndef ERROR_CUH
#define ERROR_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "ErrorType.cpp"

class Error {
	public:

		__host__ Error();

		__device__ __host__ double compute(double output, double preferredValue, ErrorType et);
		__device__ __host__ double derive(double output, double preferredValue, ErrorType et);

};

#endif