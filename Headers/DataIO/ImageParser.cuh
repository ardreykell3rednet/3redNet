#ifndef IMAGE_PARSER_CUH
#define IMAGE_PARSER_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <string>
#include <vector>


class ImageParser {

public:

	bool requiresReading = false;

	__device__ __host__ ImageParser();
	__device__ __host__ ImageParser(std::string parse_file);

	__device__ __host__ int read();
	__device__ __host__ int read_next(int number);

	__device__ __host__ std::vector<std::string> get_next_input(int z, int channel);

	__device__ __host__ std::string get_parse_file();
	__device__ __host__ void set_parse_file(std::string parse_file);

	__device__ __host__ void clear();

private:

	std::vector<std::string> inputs;

	std::string parse_file;

	int file_location = 0;

	int num_elements = 0;




};

#endif