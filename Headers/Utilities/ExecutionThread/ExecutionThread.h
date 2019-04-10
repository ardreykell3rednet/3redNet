#ifndef EXECUTION_THREAD_H
#define EXECUTION_THREAD_H

#include <thread>
#include <ctime>
#include <iostream>

#include "NeuralNetwork.cuh"

#include "DataTextFileManager.h"
#include "NeuralNetworkFileManager.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class ExecutionThread {
public:

	bool begin_load = true;
	bool begin_push = false;
	bool begin_execution = false;
	bool prep_save = false;
	bool begin_write = false;


	bool load_complete = false;
	bool push_complete = false;
	bool execution_complete = false;
	bool execution_half_complete = false;
	bool save_complete = false;
	bool write_complete = false;

	bool data_file_complete = false;

	bool finished = false;

	ExecutionThread();
	ExecutionThread(std::string data_path, std::string network_path, NeuralNetwork to_manage, Dimension input);
	ExecutionThread(std::string data_path, std::string network_path, std::string name, Dimension input);

	NeuralNetwork get_network();
	void set_network(NeuralNetwork n_manage);
	void load_data(int image);
	void push_data();
	void execute(int iter);
	void execute_manual();
	void write();
	void save();

	void set_channel_focus(int channel_focus);
	void set_execution_device(int device);

	void run_manual();

	void run(int image);
	void stop();

	void set_save();
	void set_execution();

	double* det_pref(int index);

private:

	NeuralNetwork to_manage;

	NeuralNetworkFileManager manage_nnfm;

	DataTextFileManager manage_input_dat;
	DataTextFileManager manage_output_dat;
	DataTextFileManager network_outputs;

	Dimension input_space;

	std::vector<double*> input;
	std::vector<Vector*> outputs_per_input;

	int channel_focus = -1;

	int chnl_tot = 0, t_tot = 0, z_tot = 0;
	int curr_c = 0, curr_t = 0, curr_z = 0;

	int start_x = 0, start_y = 0;

	int curr_img_index = 0;

	cudaStream_t stream;
	int execution_device;


};

#endif
