#include "InputCreator.cuh"

#include <string>

#define REPDIM 10

namespace InputCreator {
	__device__ __host__ double* create(std::string* parsed_input, int size, Dimension to_fill) {

		double* ret = new double[to_fill.width * to_fill.depth * to_fill.height];

		for (int i = 0; i < to_fill.width * to_fill.height * to_fill.depth; i++) {
			ret[i] = 0;
		}

		Vector center = { to_fill.width / 2, to_fill.height / 2, to_fill.depth / 2 };

		//printf("Center (%i, %i, %i)\n", center.x, center.y, center.z);

		for (int i = 0; i < size; i++) {
			Vector to_add = { 0, 0, 0 };

			size_t pos = 0;

			bool is_carbon = false;
			bool is_hydrogen = false;

			int del = 0;
			if (parsed_input[i].find("C") == std::string::npos)
				while ((pos = parsed_input[i].find(",")) != std::string::npos) {
					std::string token = parsed_input[i].substr(0, pos);

					if (del == 1) {
						if (token.compare("6") == 0) {
							is_carbon = true;
						}
						else if (token.compare("1") == 0) {
							is_hydrogen = true;
						}
					}

					std::string::size_type sz;

					if (del == 3) {
						double x = std::stod(parsed_input[i], &sz);

						//printf("X: %f\n", x);

						double ratio = center.x / REPDIM;

						x *= ratio;

						to_add.x = (int)(x + 0.5);

					}

					if (del == 4) {
						double y = std::stod(parsed_input[i], &sz);

						//printf("Y: %f\n", y);

						double ratio = center.y / REPDIM;

						y *= ratio;

						to_add.y = (int)(y + 0.5);


					}

					if (del == 5) {
						double z = std::stod(parsed_input[i], &sz);

						//printf("Z: %f\n", z);

						double ratio = center.z / REPDIM;

						z *= ratio;

						to_add.z = (int)(z + 0.5);

					}

					parsed_input[i].erase(0, parsed_input[i].find(",") + std::string(",").length());

					del++;
				}

			if (parsed_input[i].find("C") == std::string::npos) {
				Vector fin = to_add + center;

				int size = fin.x + fin.y * to_fill.width + fin.z * to_fill.width * to_fill.height;

				if(size < to_fill.depth * to_fill.width * to_fill.height)	
					if (is_carbon) {
						ret[size] = 6;
					}
					else if (is_hydrogen) {
						ret[size] = 1;
					}

				//printf("Vector (%i, %i, %i) :> Value %f\n", fin.x, fin.y, fin.z, ret[size]);
			}

			
		}


		return ret;
	}

	__device__ __host__ double* create_image(std::string parsed_input, Dimension to_fill) {
		
		//printf("Method Entry");
		
		double* ret = new double[to_fill.width * to_fill.depth * to_fill.height];

		int pos = 0;

		int i = 0;
		int cnt = 0;
		
		//printf("Process Start:>");

		while ((pos = parsed_input.find(",")) != std::string::npos) {
			std::string token = parsed_input.substr(0, pos);
			cnt++;

			std::string::size_type sz;

			if (cnt >= 2) {
				ret[i++] = std::stod(token, &sz)/255.0;
				
				//printf("%f,", ret[i++]);
			
			}


			parsed_input.erase(0, parsed_input.find(",") + std::string(",").length());

		}

		return ret;
	}



};