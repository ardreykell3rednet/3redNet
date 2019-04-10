#ifndef INPUT_GENERATOR_CUH
#define INPUT_GENERATOR_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <string>

class InputGenerator {

	public:
		__device__ __host__ InputGenerator();
		__device__ __host__ InputGenerator(std::string n_file_name);


		__device__ __host__ std::string get_file_name();
		__device__ __host__ void set_file_name(std::string n_file_name);

		__device__ __host__ int create();

		__device__ __host__ int write(int num_carbons, int num_hydrogens, double max_x_range, double max_y_range, double max_z_range);
		__device__ __host__ int generate(int number, int num_carbons, int num_hydrogens, double max_x_range, double max_y_range, double max_z_range);

	private:

		std::string n_file_name;


};


#endif

