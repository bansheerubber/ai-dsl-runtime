extern "C" {
	void airt_init();

	void airt_register_function(const char* name, unsigned int input_count, unsigned int output_count);

	// should be called whenever a function that has an associated neural network is called. the runtime will forward the
	// inputs through the neural network and cache those results so they can be later accessed by the airt_predict_*
	// functions
	unsigned int airt_handle_function_call(const char* function_name, float* inputs);
	void airt_finish_function_call(const char* function_name, unsigned int predict_index);

	int airt_predict_int(const char* function_name, unsigned int predict_index, unsigned int output_index);
	float airt_predict_float(const char* function_name, unsigned int predict_index, unsigned int output_index);
}
