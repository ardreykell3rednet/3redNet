#include "NeuralLayer.cuh"

#include "LayerSettings.cuh"
#include "Sigmoid.cuh"

#include "ConnectionFactory.cuh"
#include "NetworkExecutionKernel.cuh"

#include "MeanSquaredError.cuh"

#include "ActivationType.cpp"
#include "ErrorType.cpp"

#include "InputParser.cuh"
#include "InputGenerator.cuh"
#include "InputCreator.cuh"

#include "OutputWriter.cuh"

#include <iostream>

#include <ctime>

#define DIMX 16
#define DIMY 16
#define DIMZ 16

#define NUM_LAYERS 8
#define NUM_FILES 10

#define ITERATIONS 10000


double** optimized = new double*[NUM_FILES];

int carbons[NUM_FILES] = { 1,2,3,4,5,6,7,8,9,10 };
int hydrogens[NUM_FILES] = { 4,6,8,10,12,14,16,18,20,22 };
std::string files[NUM_FILES] = { "data\\ch4.csv", "data\\c2h6.csv", "data\\c3h8.csv", "data\\c4h10.csv", "data\\c5h12.csv", "data\\c6h14.csv", "data\\c7h16.csv", "data\\c8h18.csv", "data\\c9h20.csv", "data\\c10h22.csv" };

std::string output_file = "MODEL_OUTPUT_10x10x10_RES.csv";

int file_nums[ITERATIONS];

InputParser parsers[NUM_FILES];

NeuralLayer* network;
NetworkExecutionKernel nem;

InputGenerator ig;

OutputWriter ow;

void init();
void prepare_input();

int main() {

	srand(time(NULL));

	init();

	int iteration = 0;

	Error n_err = Error();

	printf("Prepping input.");

	prepare_input();

	for(int i = 0; i < ITERATIONS; i++){

		int file = file_nums[i];

		double* preferredZ = optimized[file];

		printf("Iteration %i: (C = %i, H = %i)", iteration, carbons[file], hydrogens[file]);

		clock_t start = std::clock();

		double* output = nem.network_exec(network, ErrorType::MSE, NUM_LAYERS, preferredZ);

		iteration++;

		double seconds = std::clock() / (double)CLOCKS_PER_SEC - start / (double)CLOCKS_PER_SEC;

		if (iteration % 100 == 0) {
			nem.network_apply(NUM_LAYERS);
			ow.write_next();
			ow.clear();
		}

		

		double error = 0.0;
		Error calc = Error();

		//ow.push_next("Epoch, Carbons, Hydrogens, Error, Output\n");

		std::string output_string = "";

		output_string += "" + std::to_string(i);
		output_string += ",";
		output_string += "" + std::to_string(carbons[file]);
		output_string += ",";
		output_string += "" + std::to_string(hydrogens[file]);
		output_string += ",";

		for (int i = 0; i < DIMX * DIMY * DIMZ; i++) {
			error += calc.compute(output[i], preferredZ[i], ErrorType::MSE);
		}

		output_string += "" + std::to_string(error);
		output_string += ",";

		for (int i = 0; i < DIMX * DIMY * DIMZ; i++) {
			output_string += "" + std::to_string(output[i]);
			output_string += ",";
		}

		ow.push_next(output_string);

		printf("Error %f :> EXEC TIME %f\n ", error, seconds);

		//network = backward_propagate(network, MeanSquaredError(), 3, preferred(inputZ));

		//_sleep(100);

	}

	return 0;


}

void init() {

	//Dimension net[NUM_LAYERS];

	/*net[0] = Dimension({ 2, 1, 1 });

	for (int i = 1; i < NUM_LAYERS - 1; i++) {
		net[i] = Dimension({ (unsigned int) (rand() % DIMX) + 1,(unsigned int)(rand() % DIMY) + 1,(unsigned int)(rand() % DIMZ) + 1});
	}

	net[NUM_LAYERS - 1] = { 1,1,1 };*/

	/*Dimension net[NUM_LAYERS] = { {DIMX,DIMY,DIMZ},{3,1,1},{1,1,1} };

	nem = NetworkExecutionKernel();
	
	network = new NeuralLayer[NUM_LAYERS];

	LayerSettings ls = { true, false, ActivationType::SIG, {0, 0, 0}, net[1]};

	network[0] = NeuralLayer(net[0], 0, ls);

	printf("Layer %i: (%i, %i, %i)\n", 0, network[0].get_dim().width, network[0].get_dim().height, network[0].get_dim().depth);

	for (int i = 1; i < NUM_LAYERS; i++) {
		
		if (i + 1 < NUM_LAYERS) {
			ls = { true, false, ActivationType::SIG, net[i - 1], net[i + 1] };
		}
		else {
			ls = { true, false, ActivationType::SIG, net[i - 1], {0,0,0} };
		}
		network[i] = NeuralLayer(net[i], i, ls);

		printf("Layer %i: (%i, %i, %i)\n", i, network[i].get_dim().width, network[i].get_dim().height, network[i].get_dim().depth);
	
	}

	for (int i = 0; i < NUM_LAYERS - 1; i++) {

		printf("Connecting %i to %i\n", i, i + 1);

		ConnectionFactory::connect(network[i], network[i + 1]);
	}

	std::cout << "Beginning network malloc >>>" << std::endl;

	nem.malloc(network, NUM_LAYERS);
	
	std::cout << "Malloc Success" << std::endl;
	*/

	/*Dimension net[NUM_LAYERS] = { {DIMX,DIMY,DIMZ},{10,10,1},{10,10,1}, {10,10,1}, {10,10,1},{10,10,1}, {10,10,1}, {10,10,1},{10,10,1}, {1,1,1} };

	nem = NetworkExecutionKernel();

	network = new NeuralLayer[NUM_LAYERS];

	LayerSettings ls = { true, true, ActivationType::SIG, {0, 0, 0}, net[1] };

	network[0] = NeuralLayer(net[0], 0, ls);

	printf("Layer %i: (%i, %i, %i)\n", 0, network[0].get_dim().width, network[0].get_dim().height, network[0].get_dim().depth);

	for (int i = 1; i < NUM_LAYERS; i++) {

		if (i + 1 < NUM_LAYERS) {
			ls = { true, true, ActivationType::SIG, {5,5, net[i - 1].depth}, net[i + 1] };
		}
		else {
			ls = { true, true, ActivationType::SIG, {5,5,net[i - 1].depth}, {0,0,0} };
		}
		network[i] = NeuralLayer(net[i], i, ls);

		printf("Layer %i: (%i, %i, %i)\n", i, network[i].get_dim().width, network[i].get_dim().height, network[i].get_dim().depth);

	}

	for (int i = 0; i < NUM_LAYERS - 1; i++) {

		printf("Connecting %i to %i\n", i, i + 1);

		ConnectionFactory::connect(network[i], network[i + 1], ConnectionFormat::CONV);
	}

	std::cout << "Beginning network malloc >>>" << std::endl;

	nem.malloc(network, NUM_LAYERS);

	std::cout << "Malloc Success" << std::endl;*/


	for (int i = 0; i < NUM_FILES; i++) {
		parsers[i] = InputParser(files[i]);
	}

	network = new NeuralLayer[NUM_LAYERS];
	Dimension dim[NUM_LAYERS] = { {DIMX, DIMY, DIMZ}, {16,16,4}, {8,8,4}, {8,8,6}, {6,6,6}, {8,8,4}, {16,16,4}, {DIMX, DIMY, DIMZ} };

	bool create = false;
	if(create)
		for (int i = 0; i < NUM_FILES; i++) {
			ig = InputGenerator(files[i]);
			ig.generate(100000, carbons[i], hydrogens[i], 10, 10, 10);
		}

	std::string actDat = "OptimizedData.csv";
	InputParser ip = InputParser(actDat);
	
	ip.read_next(10);

	std::vector<std::string> opt = ip.get_next_input();
	optimized[0] = InputCreator::create(&opt[0], opt.size(), dim[0]);

	opt = ip.get_next_input();
	optimized[1] = InputCreator::create(&opt[0], opt.size(), dim[0]);
	
	opt = ip.get_next_input();
	optimized[2] = InputCreator::create(&opt[0], opt.size(), dim[0]);

	opt = ip.get_next_input();
	optimized[3] = InputCreator::create(&opt[0], opt.size(), dim[0]);
	
	opt = ip.get_next_input();
	optimized[4] = InputCreator::create(&opt[0], opt.size(), dim[0]);

	opt = ip.get_next_input();
	optimized[5] = InputCreator::create(&opt[0], opt.size(), dim[0]);

	opt = ip.get_next_input();
	optimized[6] = InputCreator::create(&opt[0], opt.size(), dim[0]);

	opt = ip.get_next_input();
	optimized[7] = InputCreator::create(&opt[0], opt.size(), dim[0]);

	opt = ip.get_next_input();
	optimized[8] = InputCreator::create(&opt[0], opt.size(), dim[0]);

	opt = ip.get_next_input();
	optimized[9] = InputCreator::create(&opt[0], opt.size(), dim[0]);
	

	ow = OutputWriter(output_file);
	ow.push_next("Epoch, Carbons, Hydrogens, Error, Output\n");

	/*LayerSettings ls = { true, true, false, ActivationType::ELU, {0,0,0}, {3,3,dim[0].depth} };//INPUT
	network[0] = NeuralLayer(dim[0], 0, ls);
	
	ls = { true, false, true, ActivationType::ELU, {3,3,dim[0].depth}, {2,2,1} };//CONVOLUTIONAL
	network[1] = NeuralLayer(dim[1], 1, ls);

	ls = { true, true, false, ActivationType::LIN, {2,2,1}, {3,3, dim[2].depth} };//POOL
	network[2] = NeuralLayer(dim[2], 2, ls);

	ls = { true, false, true, ActivationType::ELU, {3,3, dim[2].depth}, {2,2,1} };//CONVOLUTIONAL
	network[3] = NeuralLayer(dim[3], 3, ls);

	ls = { true, false, true, ActivationType::LIN, {2,2,1}, dim[4] };//POOL
	network[4] = NeuralLayer(dim[4], 4, ls);

	ls = { true, false, false, ActivationType::HT, dim[4], {1,1,1} };//REG
	network[5] = NeuralLayer(dim[5], 5, ls);

	ls = { true, false, false, ActivationType::LIN, {1,1,1}, {3,3, dim[6].depth} };//REV-POOL
	network[6] = NeuralLayer(dim[6], 6, ls);

	ls = { true, true, false, ActivationType::ELU, {3,3, dim[6].depth}, {1,1,1} };//CONVOLUTIONAL
	network[7] = NeuralLayer(dim[7], 7, ls);

	ls = { true, false, false, ActivationType::LIN, {1,1,1}, {3,3, dim[8].depth} };//REV-POOL
	network[8] = NeuralLayer(dim[8], 8, ls);

	ls = { true, true, false, ActivationType::ELU, {3,3, dim[8].depth}, {0,0,0} };//OUTPUT
	network[9] = NeuralLayer(dim[9], 9, ls);*/

	LayerSettings ls = { true, false, false, ActivationType::ELU, {0,0,0}, {3,3,dim[0].depth} };//INPUT
	network[0] = NeuralLayer(dim[0], 0, ls);

	ls = { true, true, false, ActivationType::ELU, {3,3, dim[0].depth}, {2,2,1} };//CONVOLUTIONAL
	network[1] = NeuralLayer(dim[1], 1, ls);

	ls = { true, false, true, ActivationType::LIN, {2,2,1}, {3,3, dim[2].depth} };//POOL
	network[2] = NeuralLayer(dim[2], 2, ls);

	ls = { true, true, false, ActivationType::ELU, {3,3, dim[2].depth}, {2,2,1} };//CONVOLUTIONAL
	network[3] = NeuralLayer(dim[3], 3, ls);

	ls = { true, false, false, ActivationType::HT, dim[3], dim[5] };//REG
	network[4] = NeuralLayer(dim[4], 4, ls);

	ls = { true, false, false, ActivationType::HT, dim[4], {1,1,1} };//REG
	network[5] = NeuralLayer(dim[5], 5, ls);

	ls = { true, false, false, ActivationType::LIN, {1,1,1}, {3,3, dim[6].depth} };//REV-POOL
	network[6] = NeuralLayer(dim[6], 6, ls);

	ls = { true, true, false, ActivationType::ELU, {3,3, dim[6].depth}, {0,0,0} };//CONVOLUTIONAL
	network[7] = NeuralLayer(dim[7], 7, ls);


	/*printf("CONV\m");
	ConnectionFactory::connect(network[0], network[1], ConnectionFormat::CONV);
	_sleep(100);
	printf("POOL\n");
	ConnectionFactory::connect(network[1], network[2], ConnectionFormat::POOL);
	_sleep(100);
	printf("CONV\n");
	ConnectionFactory::connect(network[2], network[3], ConnectionFormat::CONV);
	_sleep(100);
	printf("POOL\n");
	ConnectionFactory::connect(network[3], network[4], ConnectionFormat::POOL);
	_sleep(100);
	printf("REG\n");
	ConnectionFactory::connect(network[4], network[5], ConnectionFormat::REG);
	_sleep(100);
	printf("REV-POOL\n");
	ConnectionFactory::connect(network[5], network[6], ConnectionFormat::POOL, true);
	_sleep(100);
	printf("CONV\n");
	ConnectionFactory::connect(network[6], network[7], ConnectionFormat::CONV);
	_sleep(100);
	printf("REV-POOL\n");
	ConnectionFactory::connect(network[7], network[8], ConnectionFormat::POOL, true);
	_sleep(100);
	printf("CONV\n");
	ConnectionFactory::connect(network[8], network[9], ConnectionFormat::CONV);*/

	ConnectionFactory::connect(network[0], network[1], ConnectionFormat::CONV);
	_sleep(100);
	printf("POOL\n");
	ConnectionFactory::connect(network[1], network[2], ConnectionFormat::POOL);
	_sleep(100);
	printf("CONV\n");
	ConnectionFactory::connect(network[2], network[3], ConnectionFormat::CONV);
	_sleep(100);
	printf("POOL\n");
	ConnectionFactory::connect(network[3], network[4], ConnectionFormat::REG);
	_sleep(100);
	printf("REG\n");
	ConnectionFactory::connect(network[4], network[5], ConnectionFormat::REG);
	_sleep(100);
	printf("REV-POOL\n");
	ConnectionFactory::connect(network[5], network[6], ConnectionFormat::POOL, true);
	_sleep(100);
	printf("CONV\n");
	ConnectionFactory::connect(network[6], network[7], ConnectionFormat::CONV);

	std::cout << "Beginning network malloc >>>" << std::endl;

	nem.malloc(network, NUM_LAYERS);

	std::cout << "Malloc Success" << std::endl;


}


void prepare_input() {
	for (int i = 0; i < ITERATIONS; i++) {
		
		int file = rand() % 10;

		if (i % (ITERATIONS / 100) == 0) {
			printf(".");
			for (int i = 0; i < NUM_FILES; i++) {
				parsers[i].read_next(100);
			}
		}

		std::vector<std::string> in;

		file_nums[i] = file;

		in = parsers[file].get_next_input();

		double* inputZ = InputCreator::create(&in[0], in.size(), { DIMX, DIMY, DIMZ });

		if (i % 500 == 0) {
		
		}

		nem.input_stack.push_back(inputZ, DIMX * DIMY * DIMZ);

	}

	printf("\n");
}
