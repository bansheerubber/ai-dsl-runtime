#include <torch/torch.h>
#include <vector>

struct RolloutBuffer {
	std::vector<torch::Tensor> actions;
	std::vector<torch::Tensor> states;
	std::vector<torch::Tensor> logProbabilities;
	std::vector<double> rewards;
	std::vector<torch::Tensor> stateValues;
	std::vector<bool> isTerminals;
};
