#include "ImageParser.cuh"

#include <iostream>
#include <fstream>

__device__ __host__ ImageParser::ImageParser() {
	inputs.reserve(30);
	parse_file = "DEFAULT_INPUT.csv";
}

ImageParser::ImageParser(std::string parse_file)
{
	inputs.reserve(30);
	this->parse_file = parse_file;
}

__device__ __host__ int ImageParser::read()
{
	std::ifstream input_file;
	input_file.open(parse_file, std::ios::beg | std::ios::in);

	std::string read;

	for (int i = 0; i < file_location; i++) {
		if (input_file.good())
			getline(input_file, read);
	}

	file_location++;

	inputs.push_back(read);

	input_file.close();

	return 0;

}

__device__ __host__ int ImageParser::read_next(int number)
{
	std::ifstream input_file;
	input_file.open(parse_file, std::ios::beg | std::ios::in);

	std::string read;

	printf("Reading Next %i", number);


	for (int i = 0; i < file_location; i++) {

		if (!input_file.eof())
			getline(input_file, read);
	}

	for (int i = 0; i < number; i++) {
		if (!input_file.eof()) {
			getline(input_file, read);
			inputs.push_back(read);
		}

		if (i % 10 == 0) {
			printf(".");
		}

	}

	input_file.close();

	printf("\n");

	return 0;
}

__device__ __host__ std::vector<std::string> ImageParser::get_next_input(int z, int channel){
	std::vector<std::string> input;

	input.push_back(inputs.at((z * 4) + channel));

	return input;


}

__device__ __host__ std::string ImageParser::get_parse_file()
{
	return parse_file;
}

__device__ __host__ void ImageParser::set_parse_file(std::string parse_file)
{
	this->parse_file = parse_file;
	this->file_location = 0;
}

__device__ __host__ void ImageParser::clear() {
	inputs.clear();
}

