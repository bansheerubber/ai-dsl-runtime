#pragma once

#include <optional>
#include <string>
#include <torch/torch.h>

#include "actorCritic.h"
#include "rolloutBuffer.h"

class PPONetwork {
public:
	PPONetwork(
		std::string functionName,
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
	torch::Tensor trainAction(torch::Tensor &state);

	// forwards an input through the network, stores the result for later consumption by `getPrediction`
	uint64_t predict(/*torch::Tensor &state*/);
	float getPrediction(uint64_t predictIndex, uint64_t outputIndex);
	void finishPredict(uint64_t predictIndex);

	void update();

	void singleTrain(float reward, bool isTerminal);
	void train(float reward, bool isTerminal);

	void setActorLearningRate(float learningRate);
	void setCriticLearningRate(float learningRate);

	void save(std::string filename);
	void load(std::string filename);

private:
	float clip;
	float gamma;
	unsigned int epochs;
	RolloutBuffer buffer;

	float actionStd;

	std::string functionName;

	std::shared_ptr<ActorCritic> policy;
	std::shared_ptr<ActorCritic> oldPolicy;

	torch::optim::Adam* actorOptimizer;
	torch::optim::Adam* criticOptimizer;
	torch::nn::MSELoss loss;

	std::unordered_map<uint64_t, torch::Tensor> predictions;
};