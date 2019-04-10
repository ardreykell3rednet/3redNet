#include "OutputWriter.cuh"

#include <iostream>
#include <fstream>

__device__ __host__ OutputWriter::OutputWriter() {
	write_buffer.reserve(30);
	o_file_name = "DEFAULT_OUTPUT.csv";
}

__device__ __host__ OutputWriter::OutputWriter(std::string o_file_name){
	write_buffer.reserve(30);
	this->o_file_name = o_file_name;
}

__device__ __host__ int OutputWriter::create()
{
	std::ofstream output_file;
	output_file.open(o_file_name, std::ios::app | std::ios::out);
	output_file.close();

	return 0;

}

__device__ __host__ int OutputWriter::write_next()
{
	std::ofstream output_file;
	output_file.open(o_file_name, std::ios::app | std::ios::out);
	
	if (output_file.is_open()) {
		for (int i = 0; i < write_buffer.size(); i++) {
			std::string to_write = write_buffer.at(i);

			output_file << to_write << "\n";

		}

		output_file << std::endl;
	}

	output_file.close();

	return 0;
}

__device__ __host__ void OutputWriter::push_next(std::string to_write){
	write_buffer.push_back(to_write);
}

__device__ __host__ void OutputWriter::clear(){
	write_buffer.clear();
}

__device__ __host__ std::string OutputWriter::get_output_file()
{
	return o_file_name;
}

__device__ __host__ void OutputWriter::set_output_file(std::string o_file)
{
	o_file_name = o_file;
}
