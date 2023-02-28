#include "runtime.h"

void Runtime::addFunctionNetwork(std::string functionName, unsigned int inputCount, unsigned int outputCount) {
	std::shared_ptr<PPONetwork> network = std::make_shared<PPONetwork>(
		functionName,
		inputCount,
		outputCount,
		0.0003,
		0.001,
		0.99,
		50,
		0.2,
		0.1
	);
	this->networks.emplace(functionName, network);
}

std::shared_ptr<PPONetwork> Runtime::getNetwork(std::string functionName) {
	return this->networks[functionName];
}

void Runtime::setResetFunction(int (*resetFunction)()) {
	this->resetFunction = resetFunction;
}

void Runtime::setTickFunction(double (*tickFunction)()) {
	this->tickFunction = tickFunction;
}

void Runtime::callResetFunction() {
	this->resetFunction();
}

double Runtime::callTickFunction() {
	return this->tickFunction();
}
