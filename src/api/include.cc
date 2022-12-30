#include "include.h"
#include "../runtime.h"

static Runtime runtime;

void airt_init() {

}

void airt_register_function(const char* name, unsigned int input_count, unsigned int output_count) {
	runtime.addFunctionNetwork(name, input_count, output_count);
}

unsigned int airt_handle_function_call(const char* function_name, float* inputs) {
	std::shared_ptr<Network> network = runtime.getNetwork(function_name);
	if (!network) {
		return 0; // TODO return null value
	}

	return network->predict();
}

void airt_finish_function_call(const char* function_name, unsigned int predict_index) {
	std::shared_ptr<Network> network = runtime.getNetwork(function_name);
	if (!network) {
		return;
	}

	return network->finishPredict(predict_index);
}

int airt_predict_int(const char* function_name, unsigned int predict_index, unsigned int output_index) {
	std::shared_ptr<Network> network = runtime.getNetwork(function_name);
	if (!network) {
		return 0;
	}

	return (int)network->getPrediction(predict_index, output_index);
}

float airt_predict_float(const char* function_name, unsigned int predict_index, unsigned int output_index) {
	std::shared_ptr<Network> network = runtime.getNetwork(function_name);
	if (!network) {
		return 0.0f;
	}

	return network->getPrediction(predict_index, output_index);
}