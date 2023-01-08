#include "memory.h"

#include <algorithm>
#include <chrono>
#include <random>

void Memory::push(Transition transition) {
	this->memory.push_back(transition);
}

std::vector<Transition> Memory::sample(unsigned int batchSize) {
	static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	static auto engine = std::default_random_engine(seed);

	shuffle(this->memory.begin(), this->memory.end(), engine);

	std::vector<Transition> output;
	for (unsigned int i = 0; i < std::min(batchSize, (unsigned int)this->memory.size()); i++) {
		output.push_back(this->memory[i]);
	}

	return output;
}
