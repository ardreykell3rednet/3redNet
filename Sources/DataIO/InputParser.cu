#include "InputParser.cuh"

#include <iostream>
#include <fstream>

__device__ __host__ InputParser::InputParser() {
	inputs.reserve(30);
	parse_file = "DEFAULT_INPUT.csv";
}

InputParser::InputParser(std::string parse_file)
{
	inputs.reserve(30);
	this->parse_file = parse_file;
}

__device__ __host__ int InputParser::read()
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

__device__ __host__ int InputParser::read_next(int number)
{
	std::ifstream input_file;
	input_file.open(parse_file, std::ios::beg | std::ios::in);

	std::string read;

	for (int i = 0; i < file_location; i++) {
		
		if(!input_file.eof())
			getline(input_file, read);
	}

	int cnt = 0;

	while (cnt < number) {

		file_location++;

		if (!input_file.good()) {
			std::getline(input_file, read);
		}
		else {
			break;
		}

		if (read.find("C")) {
			cnt++;
		}

		inputs.push_back(read);
	}

	input_file.close();

	return 0;
}

__device__ __host__ std::vector<std::string> InputParser::get_next_input()
{
	std::vector<std::string> input;

	int changes = 0;

	int start = 0;
	int end = 0;

	int counter = 0;

	for (int i = 0; i < inputs.size(); i++) {

		if (inputs.at(i).find('C') != std::string::npos) {

			if (changes == 0) {
				start = i;
			}

			changes++;


		}

		if (changes == 2) {
			end = i - 1;

			break;

		}

		if (changes == 1) {
			input.push_back(inputs.at(i));
		}
	}
	if (start < end) {
		inputs.erase(inputs.begin() + start, inputs.begin() + end);
	
		requiresReading = true;
	}


	if (inputs.size() < 10) {
		requiresReading = true;
	}
	else {
		requiresReading = false;
	}
	
	return input;

}

__device__ __host__ std::string InputParser::get_parse_file()
{
	return parse_file;
}

__device__ __host__ void InputParser::set_parse_file(std::string parse_file)
{
	this->parse_file = parse_file;
}
