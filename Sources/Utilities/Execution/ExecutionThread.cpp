#include "ExecutionThread.h"
#include "Error.cuh"

#define MAX_VECTORS 10

ExecutionThread::ExecutionThread() {

}

ExecutionThread::ExecutionThread(std::string data_path, std::string network_path, NeuralNetwork to_manage, Dimension input) {
	this->to_manage = to_manage;
	this->input_space = input;

	this->manage_nnfm = NeuralNetworkFileManager(network_path, to_manage.name);
	this->manage_input_dat = DataTextFileManager(data_path, to_manage.name + "_IN");
	this->manage_output_dat = DataTextFileManager(data_path, to_manage.name + "_OUT");
	this->network_outputs = DataTextFileManager(network_path, to_manage.name + "_OUTPUTS");

	manage_input_dat.read(0);

	std::string input_line = manage_input_dat.get_read_stack().at(0);

	chnl_tot = std::stoi(input_line.substr(0, input_line.find(',')));
	input_line = input_line.erase(0, input_line.find(',') + 1);
	z_tot = std::stoi(input_line.substr(0, input_line.find(',')));
	input_line = input_line.erase(0, input_line.find(',') + 1);
	t_tot = std::stoi(input_line.substr(0, input_line.find(',')));
	input_line = input_line.erase(0, input_line.find(',') + 1);

	printf("Chnl: %i > Z: %i > T: %i\n", chnl_tot, z_tot, t_tot);

	manage_input_dat.clear_read_stack();

	int curr_dev;

	cudaGetDevice(&curr_dev);
	cudaStreamCreate(&stream);

	to_manage.set_stream(stream);

	printf("Copying %s Network to GPU %i\n", to_manage.name.c_str(), curr_dev);

	this->to_manage.malloc();

}

ExecutionThread::ExecutionThread(std::string data_path, std::string network_path, std::string name, Dimension input) {
	this->input_space = input;

	this->manage_nnfm = NeuralNetworkFileManager(network_path, name);
	this->manage_input_dat = DataTextFileManager(data_path, name + "_IN");
	this->manage_output_dat = DataTextFileManager(data_path, name + "_OUT");
	this->network_outputs = DataTextFileManager(network_path, name + "_OUTPUTS");

	manage_input_dat.read(0);

	std::string input_line = manage_input_dat.get_read_stack().at(0);

	printf("input_line %s", input_line);

	chnl_tot = std::stoi(input_line.substr(0, input_line.find(',')));
	input_line = input_line.erase(0, input_line.find(',') + 1);
	z_tot = std::stoi(input_line.substr(0, input_line.find(',')));
	input_line = input_line.erase(0, input_line.find(',') + 1);
	t_tot = std::stoi(input_line.substr(0, input_line.find(',')));
	input_line = input_line.erase(0, input_line.find(',') + 1);

	printf("Chnl: %i > Z: %i > T: %i\n", chnl_tot, z_tot, t_tot);

	manage_input_dat.clear_read_stack();

	printf("Loading network %s.", name.c_str());

	manage_nnfm.prepare_parse();			printf(".");
	manage_nnfm.read_network_file();		printf(".");
	manage_nnfm.read_network_dimensions();	printf(".");
	manage_nnfm.parse_network();			printf(".complete\n");

	this->to_manage = manage_nnfm.get_parsed();
	this->to_manage.name = name;

	int curr_dev;
	cudaGetDevice(&curr_dev);
	cudaStreamCreate(&stream);

	to_manage.set_stream(stream);

	printf("Copying %s Network to GPU %i\n", name.c_str(), curr_dev);

	this->to_manage.malloc();
}

NeuralNetwork ExecutionThread::get_network() {
	return to_manage;
}

void ExecutionThread::set_network(NeuralNetwork n_manage) {
	this->to_manage = n_manage;
}

void ExecutionThread::load_data(int image) {

	int read_index = 2;

	int i = 0;

	std::string input_line;
	int input_data_size = 0;

	printf("Reading Image %i", image);

	do {
		manage_input_dat.read_until("IMAGE_END");

		//printf("Read Loc: %i\n", read_index);
		if (image != 0) {
			input_line = manage_input_dat.get_read_stack().at(0);
		}
		else {
			input_line = manage_input_dat.get_read_stack().at(1);
		}
		//printf("Input Line: %s\n", input_line.c_str());

		input_data_size = std::stoi(input_line.substr(0, input_line.find(',')));
		input_line = input_line.erase(0, input_line.find(',') + 1);

		curr_c = std::stoi(input_line.substr(0, input_line.find(',')));
		input_line = input_line.erase(0, input_line.find(',') + 1);

		if (curr_c == channel_focus && channel_focus != -1) {
			break;
		}
		else if (channel_focus == -1) {
			break;
		}

		image++;
		i++;

		printf(", %i", image);

		manage_input_dat.clear_read_stack();

//		read_index = read_index + input_data_size + 2;
	} while (true);

	printf("Read Location %i\n", read_index);

	curr_t = std::stoi(input_line.substr(0, input_line.find(',')));
	input_line = input_line.erase(0, input_line.find(',') + 1);

	curr_z = std::stoi(input_line.substr(0, input_line.find(',')));
	input_line = input_line.erase(0, input_line.find(',') + 1);

	//manage_input_dat.clear_read_stack();

	//manage_input_dat.read_all(read_index + 1, read_index + input_data_size + 1);

	std::vector<std::string> input_lines = manage_input_dat.get_read_stack();

	image -= i;

	if (image == 0) {
		input_lines.erase(input_lines.begin());
		input_lines.erase(input_lines.begin());
	}
	else {
		input_lines.erase(input_lines.begin());
	}

	printf("Loading Vectors.");

	for (int i = 0; i < input_data_size; i++) {
		std::string to_parse = input_lines.at(i);

		start_x = std::stoi(to_parse.substr(0, to_parse.find(',')));
		to_parse = to_parse.erase(0, to_parse.find(',') + 1);

		start_y = std::stoi(to_parse.substr(0, to_parse.find(',')));
		to_parse = to_parse.erase(0, to_parse.find(',') + 1);

		//printf("Start Loc: (%i, %i)\n", start_x, start_y);

		int num_vectors = std::stoi(to_parse.substr(0, to_parse.find(',')));
		to_parse = to_parse.erase(0, to_parse.find(',') + 1);

		Vector* vec_alloc = new Vector[MAX_VECTORS];

		//printf("Vectors Found: ");

		for (int v = 0; v < num_vectors; v++) {

			Vector to_add = { 0,0,0 };

			to_add.x = std::stod(to_parse.substr(0, to_parse.find(',')));
			to_parse = to_parse.erase(0, to_parse.find(',') + 1);

			to_add.y = std::stod(to_parse.substr(0, to_parse.find(',')));
			to_parse = to_parse.erase(0, to_parse.find(',') + 1);

			to_add.z = curr_z;

			vec_alloc[v] = to_add;

			//printf("{%i, %i, %i}, ", to_add.x, to_add.y, to_add.z);

		}

		for (int v = num_vectors; v < MAX_VECTORS; v++) {
			Vector to_add = { -1,-1, curr_z };

			to_parse = to_parse.erase(0, to_parse.find(',') + 1);
			to_parse = to_parse.erase(0, to_parse.find(',') + 1);

			vec_alloc[v] = to_add;

		}

		outputs_per_input.push_back(vec_alloc);

		int num_img_vals = input_space.width * input_space.height * input_space.depth;

		double* img_vals = new double[num_img_vals];

		//printf("Parse: %s\n", to_parse.c_str());
		for (int j = 0; j < num_img_vals; j++) {

			try {
				img_vals[j] = std::stod(to_parse.substr(0, to_parse.find(',')));
			}
			catch (std::exception &e) {
				img_vals[j] = 0;
			}

			to_parse = to_parse.erase(0, to_parse.find(',') + 1);
		}

		input.push_back(img_vals);

		if (i % 300 == 0) {
			printf(".");
		}

	}

	printf("complete\n");

	load_complete = true;
}

void ExecutionThread::push_data() {

	Dimension wrap = to_manage.get_layer(0).get_dim();

	printf("Wrap = {%i, %i, %i}: Beginning push....", wrap.width, wrap.height, wrap.depth);

	for (int i = 0; i < input.size(); i++) {

		to_manage.nem.push(input.at(i), wrap.width * wrap.height * wrap.depth);

		if (i % 400 == 0) {
			printf(".");
		}

	}

	printf("complete\n");

	push_complete = true;
}

void ExecutionThread::execute(int i) {

	Dimension fin_dim = to_manage.get_layer(to_manage.layers - 1).get_dim();
	Dimension wrap_dim = to_manage.get_layer(0).get_dim();

	execution_complete = false;

	printf("%s Iteration %i:", to_manage.name.c_str(), i);
	double* preferred = det_pref(i);

	std::string pref_string = "";
	pref_string += "" + std::to_string(i) + ",0.0,";

	for (int j = 0; j < fin_dim.width * fin_dim.height; j++) {
		pref_string += "" + std::to_string(preferred[j]);
		pref_string += ",";
	}

	manage_output_dat.push_write(pref_string);

	to_manage.execute(preferred);

	execution_complete = true;

	double* output = to_manage.output;

	double error = 0.0;
	Error calc = Error();

	std::string output_string = "";

	output_string += "" + std::to_string(i);
	output_string += ",";

	for (int j = 0; j < fin_dim.width * fin_dim.height; j++) {
		error += calc.compute(output[j], preferred[j], ErrorType::MSE);
	}

	printf("  Error - %f :> complete\n", error);

	cudaDeviceSynchronize();

	output_string += "" + std::to_string(error);
	output_string += ",";

	for (int j = 0; j < fin_dim.width * fin_dim.height; j++) {
		output_string += "" + std::to_string(output[j]);
		output_string += ",";
	}

	manage_output_dat.push_write(output_string);
	

	curr_img_index++;

}

void ExecutionThread::execute_manual() {

	Dimension fin_dim = to_manage.get_layer(to_manage.layers - 1).get_dim();
	Dimension wrap_dim = to_manage.get_layer(0).get_dim();

	for (int i = 0; i < input.size(); i++) {

		to_manage.execute();

		double* output = to_manage.output;

		std::string output_string = "";

		output_string += "" + std::to_string(i);
		output_string += ",";

		if (i == input.size() / 2) {
			execution_half_complete = true;
		}

		for (int j = 0; j < fin_dim.width * fin_dim.height; j++) {
			output_string += "" + std::to_string(output[j]);
			output_string += ",";
		}

		manage_output_dat.push_write(output_string);
	}

	curr_img_index++;

	execution_complete = true;
}

void ExecutionThread::write() {

	printf("Writing Data... \n");

	manage_output_dat.write_stack();

	for (int i = 0; i < input.size(); i++) {
		delete[] input.at(i);
		delete[] outputs_per_input.at(i);
	}

	input.clear();
	outputs_per_input.clear();

	write_complete = true;

}

void ExecutionThread::save() {

	to_manage.save();

	manage_nnfm.set_to_write(to_manage);

	manage_nnfm.prepare_write();
	manage_nnfm.write_network_file();
	manage_nnfm.write_network_dimensions();
	manage_nnfm.write_network();

	save_complete = true;

}

void ExecutionThread::set_channel_focus(int channel_focus) {
	this->channel_focus = channel_focus;
}

void ExecutionThread::set_execution_device(int device) {
	this->execution_device = device;
}

void ExecutionThread::run_manual() {
	if (begin_load) {
		std::thread ld(&ExecutionThread::load_data, this, 1);
		begin_load = false;
	}

	if (begin_push) {
		std::thread pd(&ExecutionThread::push_data, this);
		begin_push = false;
	}

	if (begin_execution) {
		std::thread ex(&ExecutionThread::execute_manual, this);
		begin_execution = false;
	}

	if (prep_save) {
		std::thread s(&ExecutionThread::save, this);
		return;
	}

	if (begin_write) {
		std::thread wr(&ExecutionThread::write, this);
		begin_write = false;
	}

	if (load_complete) {
		begin_load = false;
		begin_push = true;
		load_complete = false;
	}

	if (push_complete) {
		begin_push = false;
		begin_execution = true;
		push_complete = false;
	}

	if (execution_complete) {
		begin_execution = false;
		begin_write = true;
		execution_complete = false;
	}

	if (execution_half_complete) {
		begin_load = true;
		execution_half_complete = false;
	}

	if (curr_img_index == chnl_tot * t_tot * z_tot - 1) {
		data_file_complete = true;
		return;
	}
}

void ExecutionThread::run(int image) {

	finished = false;

	manage_output_dat.push_write("IMAGE_START,\n");

	load_data(image);

	printf("Load Finished\n");

	push_data();

	for (int i = 0; i < input.size(); i++) {

		while (!begin_execution) {}
		execute(i);

		begin_execution = false;
	}

	manage_output_dat.push_write("IMAGE_END,\n");

	write();

	if (curr_img_index == chnl_tot / 2 * t_tot * z_tot - 1) {
		data_file_complete = true;
		return;
	}

	to_manage.apply();

	finshed = true;

}

void ExecutionThread::stop() {
	prep_save = true;
}

void ExecutionThread::set_save() {
	prep_save = true;
}

void ExecutionThread::set_execution() {
	begin_execution = true;
}

double* ExecutionThread::det_pref(int index) {

	Dimension fin_dim = to_manage.get_layer(to_manage.layers - 1).get_dim();
	Dimension wrap_dim = to_manage.get_layer(0).get_dim();

	Vector* vectors = outputs_per_input.at(index);
	double* preferred = new double[fin_dim.width * fin_dim.height * fin_dim.depth];

	for (int x = 0; x < fin_dim.width; x++) {
		for (int y = 0; y < fin_dim.height; y++) {

			if (x == 0) {
				if (vectors[y].x > 1) {
					preferred[x + y * fin_dim.width] = vectors[y].x - start_x;
				}
				else {
					preferred[x + y * fin_dim.width] = -1;
				}
			}
			else if (x == 1) {
				if (vectors[y].x > 1) {
					preferred[x + y * fin_dim.width] = vectors[y].y - start_y;
				}
				else {
					preferred[x + y * fin_dim.width] = -1;
				}
			}
			else {
				preferred[x + y * fin_dim.width] = 1.0;
			}

		}

	}

	return preferred;

}

