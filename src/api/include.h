#include <stdint.h>

extern "C" {
	void airt_init(int (*reset_function)(), double (*tick_function)());

	void airt_register_function(const char* name, uint64_t input_count, uint64_t output_count);

	// should be called whenever a function that has an associated neural network is called. the runtime will forward the
	// inputs through the neural network and cache those results so they can be later accessed by the airt_predict_*
	// functions
	uint64_t airt_handle_function_call(const char* function_name, double* inputs);
	void airt_finish_function_call(const char* function_name, uint64_t predict_index);

	int64_t airt_predict_int(const char* function_name, uint64_t predict_index, uint64_t output_index);
	double airt_predict_float(const char* function_name, uint64_t predict_index, uint64_t output_index);

	void airt_train();

	void _airt_print_float(double number);
	void _airt_print_int(uint16_t number);

	double _airt_random_float(double min, double max);
	void _airt_log_simulation(double car1_position, double car2_position);
}
