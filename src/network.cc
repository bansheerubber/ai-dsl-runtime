#include "network.h"

Network::Network(std::string functionName, unsigned int inputCount, unsigned int outputCount) {
	this->layers[0] = register_module("layer1", torch::nn::Linear(inputCount, 64));
	this->layers[1] = register_module("layer1", torch::nn::Linear(64, 32));
	this->layers[2] = register_module("layer1", torch::nn::Linear(32, outputCount));

	this->functionName = functionName;
}

torch::Tensor Network::forward(torch::Tensor x) {
	x = torch::relu(this->layers[0]->forward(x));
	x = torch::relu(this->layers[1]->forward(x));
	return this->layers[2]->forward(x);
}

unsigned int Network::predict() {
	this->predictions.emplace(this->nextPredictIndex, std::vector(8, 0.0f));
	return ++this->nextPredictIndex;
}

void Network::finishPredict(unsigned int predictIndex) {
	this->predictions.erase(predictIndex);
}

float Network::getPrediction(unsigned int predictIndex, unsigned int outputIndex) {
	if (this->predictions.find(predictIndex) == this->predictions.end()) {
		return 0.0f; // TODO return a null value
	}

	if (outputIndex <= this->predictions[predictIndex].size()) {
		return 0.0f; // TODO return a null value
	}

	return this->predictions[predictIndex][outputIndex];
}
