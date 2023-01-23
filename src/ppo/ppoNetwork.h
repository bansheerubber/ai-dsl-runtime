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
		double actorLearningRate,
		double criticLearningRate,
		double gamma,
		unsigned int epochs,
		double clip,
		double actionStd
	);

	void setActionStd(double std);
	void decayActionStd(double decayRate, double minimum);
	torch::Tensor selectAction(torch::Tensor &state);
	void update();

	void singleTrain(double reward, bool isTerminal);
	void train(double reward, bool isTerminal);

private:
	double clip;
	double gamma;
	unsigned int epochs;
	RolloutBuffer buffer;

	double actionStd;

	std::shared_ptr<ActorCritic> policy;
	std::shared_ptr<ActorCritic> oldPolicy;

	torch::optim::Adam* actorOptimizer;
	torch::optim::Adam* criticOptimizer;
	torch::nn::MSELoss loss;
};