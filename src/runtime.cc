#include "runtime.h"

void Runtime::addFunctionNetwork(std::string functionName, unsigned int inputCount, unsigned int outputCount) {
	std::shared_ptr<DQNNetwork> network = std::make_shared<DQNNetwork>(functionName, inputCount, outputCount);
	this->networks.emplace(functionName, network);
}

std::shared_ptr<DQNNetwork> Runtime::getNetwork(std::string functionName) {
	return this->networks[functionName];
}
