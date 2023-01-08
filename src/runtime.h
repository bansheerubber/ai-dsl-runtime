#pragma once

#include <unordered_map>

#include "dqnNetwork.h"

// state that keeps track of what is currently happening in the language runtime
class Runtime {
public:
	void addFunctionNetwork(std::string functionName, unsigned int inputCount, unsigned int outputCount);
	std::shared_ptr<DQNNetwork> getNetwork(std::string functionName);

private:
	// map function names to their neural networks
	std::unordered_map<std::string, std::shared_ptr<DQNNetwork>> networks;
};
