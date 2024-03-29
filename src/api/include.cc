#include "include.h"

#include <ctype.h>
#include <random>

#include "../runtime.h"

static Runtime runtime;

void airt_init(int (*reset_function)(), double (*tick_function)()) {
	std::cout << "airt initialized" << std::endl;
	runtime.setResetFunction(reset_function);
	runtime.setTickFunction(tick_function);
}

void airt_register_function(const char* name, uint64_t input_count, uint64_t output_count) {
	runtime.addFunctionNetwork(name, input_count, output_count);
}

uint64_t airt_handle_function_call(const char* function_name, double* inputs) {
	std::shared_ptr<PPONetwork> network = runtime.getNetwork(function_name);
	if (!network) {
		return 0; // TODO return null value
	}

	return network->predict(inputs);
}

void airt_finish_function_call(const char* function_name, uint64_t predict_index) {
	std::shared_ptr<PPONetwork> network = runtime.getNetwork(function_name);
	if (!network) {
		return;
	}

	return network->finishPredict(predict_index);
}

int64_t airt_predict_int(const char* function_name, uint64_t predict_index, uint64_t output_index) {
	std::shared_ptr<PPONetwork> network = runtime.getNetwork(function_name);
	if (!network) {
		return 0;
	}

	return (int)network->getPrediction(predict_index, output_index);
}

double airt_predict_float(const char* function_name, uint64_t predict_index, uint64_t output_index) {
	std::shared_ptr<PPONetwork> network = runtime.getNetwork(function_name);
	if (!network) {
		return 0.0f;
	}

	return network->getPrediction(predict_index, output_index);
}

void airt_train() {
	runtime.train();
}

void _airt_print_float(double number) {
	std::cout << "double: " << number << std::endl;
}

void _airt_print_int(uint16_t number) {
	std::cout << "int: " << number << std::endl;
}

double _airt_random_float(double min, double max) {
	static std::random_device rd;
	static std::mt19937 e2(rd());

	std::uniform_real_distribution<> dist(min, max);
	return dist(e2);
}

void _airt_log_simulation(double car1_position, double car2_position) {
	runtime.logSimulation(car1_position, car2_position);
}
