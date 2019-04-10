#ifndef OUTPUT_WRITER_CUH
#define OUTPUT_WRITER_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <string>
#include <vector>

class OutputWriter {

	public:
		__device__ __host__ OutputWriter();
		__device__ __host__ OutputWriter(std::string o_file_name);

		__device__ __host__ int create();

		__device__ __host__ int write_next();

		__device__ __host__ void push_next(std::string to_write);
		__device__ __host__ void clear();

		__device__ __host__ std::string get_output_file();
		__device__ __host__ void set_output_file(std::string parse_file);

	private:
		std::vector<std::string> write_buffer;
		std::string o_file_name;

};

#endif