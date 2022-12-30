#pragma once

#include <string>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

class Network : public torch::nn::Module {
public:
	Network(std::string functionName, unsigned int inputCount, unsigned int outputCount);

	torch::Tensor forward(torch::Tensor x);

	unsigned int predict();
	void finishPredict(unsigned int predictIndex);
	float getPrediction(unsigned int predictIndex, unsigned int outputIndex);

private:
	torch::nn::Linear layers[3] = { nullptr, nullptr, nullptr };

	std::string functionName;

	unsigned int nextPredictIndex = 0;
	std::unordered_map<unsigned int, std::vector<float>> predictions;
};
