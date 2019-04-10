#include "InputGenerator.cuh"

#include <iostream>
#include <fstream>

#define RES 10000

__device__ __host__ InputGenerator::InputGenerator() {
	n_file_name = "DEFAULT_INPUT.csv";
}

InputGenerator::InputGenerator(std::string n_file_name){

	this->n_file_name = n_file_name;

}

__device__ __host__ std::string InputGenerator::get_file_name()
{
	return n_file_name;
}

__device__ __host__ void InputGenerator::set_file_name(std::string n_file_name)
{
	this->n_file_name = n_file_name;
	create();
}

__device__ __host__ int InputGenerator::create()
{
	std::ofstream generate_file;
	generate_file.open(n_file_name, std::ios::ate | std::ios::out);
	generate_file.close();

	return 0;
}

__device__ __host__ int InputGenerator::generate(int number, int num_carbons, int num_hydrogens, double max_x_range, double max_y_range, double max_z_range)
{
	std::ofstream generate_file;
	generate_file.open(n_file_name, std::ios::ate | std::ios::out);

	for(int i = 0; i < number; i++)
		if (generate_file.is_open()) {
			generate_file << "C" << num_carbons << "H" << num_hydrogens << ",\n";

			int number = 0;

			for (int i = 0; i < num_carbons; i++) {
				generate_file << number << "," << "6," << "0,";

				double x = ((double)(rand() % RES) / RES) * max_x_range;
				double y = ((double)(rand() % RES) / RES) * max_y_range;
				double z = ((double)(rand() % RES) / RES) * max_z_range;

				if (rand() % 2 == 1) {
					x *= -1;
				}

				if (rand() % 2 == 1) {
					y *= -1;
				}

				if (rand() % 2 == 1) {
					z *= -1;
				}

				generate_file << x << "," << y << "," << z << "," << std::endl;

				number++;

			}

			for (int i = 0; i < num_hydrogens; i++) {
				generate_file << number << "," << "1," << "0,";

				double x = ((double)(rand() % RES) / RES) * max_x_range;
				double y = ((double)(rand() % RES) / RES) * max_y_range;
				double z = ((double)(rand() % RES) / RES) * max_z_range;

				if (rand() % 2 == 1) {
					x *= -1;
				}

				if (rand() % 2 == 1) {
					y *= -1;
				}

				if (rand() % 2 == 1) {
					z *= -1;
				}

				generate_file << x << "," << y << "," << z << "," << std::endl;

				number++;
			}
		}

	generate_file.close();

	return 0;
}

__device__ __host__ int InputGenerator::write(int num_carbons, int num_hydrogens, double max_x_range, double max_y_range, double max_z_range)
{
	std::ofstream generate_file;
	generate_file.open(n_file_name, std::ios::ate | std::ios::out);
	
	if (generate_file.is_open()) {
		generate_file << "C" << num_carbons << "H" << num_hydrogens << ",\n";

		int number = 0;

		for (int i = 0; i < num_carbons; i++) {
			generate_file << number << "," << "6," << "0,";

			double x = ((double)(rand() % RES) / RES) * max_x_range;
			double y = ((double)(rand() % RES) / RES) * max_y_range;
			double z = ((double)(rand() % RES) / RES) * max_z_range;

			if (rand() % 2 == 1) {
				x *= -1;
			}

			if (rand() % 2 == 1) {
				y *= -1;
			}

			if (rand() % 2 == 1) {
				z *= -1;
			}

			generate_file << x << "," << y << "," << z << "," << std::endl;
		}

		for (int i = 0; i < num_hydrogens; i++) {
			generate_file << number << "," << "1," << "0,";

			double x = ((double)(rand() % RES) / RES) * max_x_range;
			double y = ((double)(rand() % RES) / RES) * max_y_range;
			double z = ((double)(rand() % RES) / RES) * max_z_range;

			if (rand() % 2 == 1) {
				x *= -1;
			}

			if (rand() % 2 == 1) {
				y *= -1;
			}

			if (rand() % 2 == 1) {
				z *= -1;
			}

			generate_file << x << "," << y << "," << z << "," << std::endl;

			number++;
		}
	}

	generate_file.close();

	return 0;
}
