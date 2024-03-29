#pragma once

#include <iostream>
#include <fstream>
#include <unordered_map>

#include "ppo/ppoNetwork.h"

// state that keeps track of what is currently happening in the language runtime
class Runtime {
public:
	void addFunctionNetwork(std::string functionName, unsigned int inputCount, unsigned int outputCount);
	std::shared_ptr<PPONetwork> getNetwork(std::string functionName);

	void setResetFunction(int (*resetFunction)());
	void setTickFunction(double (*tickFunction)());

	void callResetFunction();
	double callTickFunction();

	void train();

	void logSimulation(double car1Position, double car2Position);

private:
	// map function names to their neural networks
	std::unordered_map<std::string, std::shared_ptr<PPONetwork>> networks;

	int (*resetFunction)() = nullptr;
	double (*tickFunction)() = nullptr;

	std::ofstream file;
	uint64_t fileIndex = 0;
};
