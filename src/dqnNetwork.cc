#include "dqnNetwork.h"

#include <chrono>
#include <random>

DQNNetwork::DQNNetwork(std::string functionName, unsigned int inputCount, unsigned int outputCount) {
	this->layers[0] = register_module("layer1", torch::nn::Linear(inputCount, 128));
	this->layers[1] = register_module("layer1", torch::nn::Linear(128, 128));
	this->layers[2] = register_module("layer1", torch::nn::Linear(128, outputCount));

	this->inputCount = inputCount;
	this->outputCount = outputCount;

	this->functionName = functionName;
}

torch::Tensor DQNNetwork::forward(torch::Tensor x) {
	x = torch::relu(this->layers[0]->forward(x));
	x = torch::relu(this->layers[1]->forward(x));
	return this->layers[2]->forward(x);
}

unsigned int DQNNetwork::predict() {
	this->predictions.emplace(this->nextPredictIndex, std::vector(8, 0.0f));
	return ++this->nextPredictIndex;
}

void DQNNetwork::finishPredict(unsigned int predictIndex) {
	this->predictions.erase(predictIndex);
}

float DQNNetwork::getPrediction(unsigned int predictIndex, unsigned int outputIndex) {
	if (this->predictions.find(predictIndex) == this->predictions.end()) {
		return 0.0f; // TODO return a null value
	}

	if (outputIndex <= this->predictions[predictIndex].size()) {
		return 0.0f; // TODO return a null value
	}

	return this->predictions[predictIndex][outputIndex];
}

torch::Tensor DQNNetwork::selectAction(torch::Tensor state) {
	static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	static auto engine = std::default_random_engine(seed);
	static std::uniform_real_distribution<> distribution(0.0, 1.0);
	std::uniform_int_distribution<long> actionDistribution(0, this->outputCount);

	double threshold = 0.0;
	double sample = distribution(engine);
	if (sample < threshold) {
		return std::get<1>(this->forward(state).max(1)).view({ 1, 1 });
	} else {
		return torch::tensor({{ actionDistribution(engine) }});
	}
}
