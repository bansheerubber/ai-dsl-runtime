#pragma once

#include <optional>
#include <string>
#include <torch/torch.h>

#include "actorCritic.h"
#include "rolloutBuffer.h"

class PPONetwork {
public:
	PPONetwork(
		unsigned int inputCount,
		unsigned int outputCount,
		float actorLearningRate,
		float criticLearningRate,
		float gamma,
		unsigned int epochs,
		float clip,
		float actionStd
	);

	void setActionStd(float std);
	float decayActionStd(float decayRate, float minimum);
	torch::Tensor selectAction(torch::Tensor &state);
	ActResult predict(torch::Tensor &state);
	void update();

	void singleTrain(float reward, bool isTerminal);
	void train(float reward, bool isTerminal);

private:
	float clip;
	float gamma;
	unsigned int epochs;
	RolloutBuffer buffer;

	float actionStd;

	std::shared_ptr<ActorCritic> policy;
	std::shared_ptr<ActorCritic> oldPolicy;

	torch::optim::Adam* actorOptimizer;
	torch::optim::Adam* criticOptimizer;
	torch::nn::MSELoss loss;
};