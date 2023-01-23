#pragma once

#include <string>
#include <torch/torch.h>

struct ActResult {
	torch::Tensor action;
	torch::Tensor probability;
	torch::Tensor critic;
};

struct EvaluateResult {
	torch::Tensor probability;
	torch::Tensor critic;
	torch::Tensor entropy;
};

class ActorCritic : public torch::nn::Cloneable<ActorCritic> {
public:
	ActorCritic() {}
	ActorCritic(unsigned int inputCount, unsigned int outputCount, double initStd);

	void setActionStd(double std);
	ActResult act(torch::Tensor &state);
	EvaluateResult evaluate(torch::Tensor &state, torch::Tensor &action);

	torch::nn::Sequential actor;
	torch::nn::Sequential critic;

	void reset() override;
	void copy(ActorCritic &source);

private:
	unsigned int inputCount;
	unsigned int outputCount;

	torch::Tensor actionVariance;
};
