#include "ConnectionFactory.cuh"
#include "ConnectionFormat.cpp"

#include "VectorStructs.cuh"

namespace ConnectionFactory {

	void connect(NeuralLayer from, NeuralLayer to, ConnectionFormat format, bool reverse_application, unsigned int kernel_size) {
		if (format == ConnectionFormat::CONV) {
			connect_conv(from, to, reverse_application, kernel_size);
		}
		else if (format == ConnectionFormat::RAND) {
			connect_rand(from, to, reverse_application);
		}
		else if (format == ConnectionFormat::REG) {
			connect_reg(from, to, reverse_application);
		}
		else if (format == ConnectionFormat::POOL) {
			connect_pool(from, to, reverse_application);
		}
	}

	void connect_conv(NeuralLayer from, NeuralLayer to, bool reverse_application, unsigned int kernel_size) {

		Connection* connection;

		for (unsigned int z = 0; z < to.get_dim().depth; z++) {
			for (unsigned int x = 0; x < to.get_dim().width; x++) {
				for (unsigned int y = 0; y < to.get_dim().height; y++) {

					Dimension obj_dim = { kernel_size, kernel_size, from.get_dim().depth };

					//printf("obj_dim: (%i, %i, %i)\n", obj_dim.width, obj_dim.height, obj_dim.depth);

					connection = new Connection[obj_dim.height * obj_dim.width * obj_dim.depth];

					for (unsigned int i = 0; i < kernel_size; i++)
						for (unsigned int j = 0; j < kernel_size; j++) {
							for (unsigned int nZ = 0; nZ < from.get_dim().depth; nZ++) {


								LayerLocation f = { {x - kernel_size / 2 + i, y - kernel_size / 2 + j, nZ}, from.get_id() };
								LayerLocation t = { { x, y, z}, to.get_id() };


								Connection to_add = { f,t };

								//from.get_weights_out_of(f.location).set_connection(to_add, t.location);

								connection[i + j * obj_dim.width + nZ * obj_dim.width * obj_dim.height] = to_add;
							}
						}

					to.get_weights_in_of(x, y, z).reinitialize(connection);

					delete[] connection;
				}
			}
		}

	}

	void connect_reg(NeuralLayer from, NeuralLayer to, bool reverse_application) {

		Connection* connection;

		for (unsigned int z = 0; z < to.get_dim().depth; z++) {
			for (unsigned int x = 0; x < to.get_dim().width; x++) {
				for (unsigned int y = 0; y < to.get_dim().height; y++) {
					
					Dimension obj_dim = to.get_obj_in_dim();

					//printf("obj_dim: (%i, %i, %i)\n", obj_dim.width, obj_dim.height, obj_dim.depth);

					connection = new Connection[obj_dim.height * obj_dim.width * obj_dim.depth];

					for (unsigned int bz = 0; bz < obj_dim.depth; bz++) {
						for (unsigned int bx = 0; bx < obj_dim.width; bx++) {
							for (unsigned int by = 0; by < obj_dim.height; by++) {


								LayerLocation f = { {bx, by, bz}, from.get_id() };
								LayerLocation t = { { x, y, z}, to.get_id() };

								Connection to_add = { f,t };

								//from.get_weights_out_of(f.location).set_connection(to_add, t.location);

								connection[bx + by * obj_dim.width + bz * obj_dim.width * obj_dim.height] = to_add;

							}
						}
					}

					to.get_weights_in_of(x, y, z).reinitialize(connection);

					delete[] connection;

				}
			}
		}

		//delete connection;

	}

	void connect_rand(NeuralLayer from, NeuralLayer to, bool reverse_application) {

		Dimension weightDim = to.get_obj_in_dim();

		std::vector<Connection> connection;

		for (unsigned int z = 0; z < to.get_dim().depth; z++) {
			for (unsigned int x = 0; x < to.get_dim().width; x++) {
				for (unsigned int y = 0; y < to.get_dim().height; y++) {

					std::vector<Vector> usedNumbers;

					int usedSize = 0;

					int randX = rand() % weightDim.width;
					int randY = rand() % weightDim.height;
					int randZ = rand() % weightDim.depth;

					for (unsigned int bz = 0; bz < randZ; bz++)
						for (unsigned int bx = 0; bx < randX; bx++)
							for (unsigned int by = 0; by < randY; by++) {

								bool equal = false;

								Vector toTest = { rand() % (int) from.get_dim().width, rand() % (int) from.get_dim().height, rand() % (int) from.get_dim().depth };

								do {

									for (int i = 0; i < usedSize; i++) {
										if (usedNumbers[i].x == toTest.x && usedNumbers[i].y == toTest.y && usedNumbers[i].z == toTest.z) {
											equal = true;
										}
									}

									if (!equal) { break; }

									toTest = Vector({ rand() % (int) from.get_dim().width, rand() % (int) from.get_dim().height, rand() % (int) from.get_dim().depth });

								} while (true);

								LayerLocation f = { {toTest.x, toTest.y, toTest.z}, from.get_id() };
								LayerLocation t = {{ x, y, z}, to.get_id()};

								Connection toPush = { f,t };

								//from.get_weights_out_of(f.location).set_connection(toPush, t.location);

								connection.push_back(toPush);

								usedNumbers.push_back(toTest);
								usedSize++;
							}

					to.get_weights_in_of(x, y, z).reinitialize(connection);

				}
			}
		}

	}



	void connect_pool(NeuralLayer from, NeuralLayer to, bool reverse_application) {
		if (!reverse_application) {
			Connection* connection;

			for (unsigned int z = 0; z < to.get_dim().depth; z++) {
				for (unsigned int x = 0; x < to.get_dim().width; x++) {
					for (unsigned int y = 0; y < to.get_dim().height; y++) {

						connection = new Connection[2 * 2];

						/*for (unsigned int bx = 0; bx < obj_dim.width; bx++) {
							for (unsigned int by = 0; by < obj_dim.height; by++) {


								LayerLocation f = { {bx, by, bz}, from.get_id() };
								LayerLocation t = { { x, y, z}, to.get_id() };

								Connection to_add = { f,t };

								//from.get_weights_out_of(f.location).set_connection(to_add, t.location);

								connection[bx + by * obj_dim.width + bz * obj_dim.width * obj_dim.height] = to_add;

							}
						}*/

						for (int i = 0; i < 2; i++) {
							for (int j = 0; j < 2; j++) {
								LayerLocation f = { {x * 2 + i, y * 2 + j, z}, from.get_id() };
								LayerLocation t = { { x, y, z}, to.get_id() };

								Connection to_add = { f,t };

								connection[i * 2 + j] = to_add;
							}
						}


						to.get_weights_in_of(x, y, z).reinitialize(connection);

						delete[] connection;



					}
				}
			}
		}
		else {
			Connection* connection;

			for (unsigned int z = 0; z < to.get_dim().depth; z++) {
				for (unsigned int x = 0; x < to.get_dim().width; x++) {
					for (unsigned int y = 0; y < to.get_dim().height; y++) {

						Dimension obj_dim = from.get_dim();

						connection = new Connection[1 * 1];

						/*for (unsigned int bx = 0; bx < obj_dim.width; bx++) {
							for (unsigned int by = 0; by < obj_dim.height; by++) {


								LayerLocation f = { {bx, by, bz}, from.get_id() };
								LayerLocation t = { { x, y, z}, to.get_id() };

								Connection to_add = { f,t };

								//from.get_weights_out_of(f.location).set_connection(to_add, t.location);

								connection[bx + by * obj_dim.width + bz * obj_dim.width * obj_dim.height] = to_add;

							}
						}*/

						LayerLocation f = { {(int)x / 2, (int)y / 2, z}, from.get_id() };
						LayerLocation t = { { x, y, z}, to.get_id() };

						Connection to_add = { f,t };

						connection[0] = to_add;



						to.get_weights_in_of(x, y, z).reinitialize(connection);

						delete[] connection;
					}
				}
			}
		}
	}
}

