#include "runtime.h"

void Runtime::addFunctionNetwork(std::string functionName, unsigned int inputCount, unsigned int outputCount) {
	std::shared_ptr<Network> network = std::make_shared<Network>(functionName, inputCount, outputCount);
	this->networks.emplace(functionName, network);
}

std::shared_ptr<Network> Runtime::getNetwork(std::string functionName) {
	return this->networks[functionName];
}
