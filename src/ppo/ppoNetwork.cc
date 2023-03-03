#include "ppoNetwork.h"

PPONetwork::PPONetwork(
	std::string functioName,
	unsigned int inputCount,
	unsigned int outputCount,
	float actorLearningRate,
	float criticLearningRate,
	float gamma,
	unsigned int epochs,
	float clip,
	float actionStd
) {
	this->gamma = gamma;
	this->clip = clip;
	this->epochs = epochs;

	this->actionStd = actionStd;

	this->functionName = functionName;

	this->inputCount = inputCount;
	this->outputCount = outputCount;

	this->policy = std::make_shared<ActorCritic>(inputCount, outputCount, actionStd, false);
	this->policy->to(torch::Device(torch::kCUDA, 0));

	this->actorOptimizer = new torch::optim::Adam(
		this->policy->actor.get()->parameters(), torch::optim::AdamOptions(actorLearningRate)
	);

	this->criticOptimizer = new torch::optim::Adam(
		this->policy->critic.get()->parameters(), torch::optim::AdamOptions(criticLearningRate)
	);

	this->oldPolicy = std::dynamic_pointer_cast<ActorCritic>(this->policy->clone());
	this->oldPolicy->setCpuOnly(true);

	this->buffer.unrewardedActions = 0;
}

void PPONetwork::setActionStd(float std) {
	this->actionStd = std;
	this->policy->setActionStd(std);
	this->oldPolicy->setActionStd(std);
}

float PPONetwork::decayActionStd(float decayRate, float minimum) {
	this->actionStd = this->actionStd - decayRate;
	// TODO round action STD

	if (this->actionStd < minimum) {
		this->actionStd = minimum;
	}

	this->setActionStd(this->actionStd);

	return this->actionStd;
}

torch::Tensor PPONetwork::trainAction(torch::Tensor &state) {
	ActResult act;
	{
		torch::NoGradGuard no_grad;
		act = this->oldPolicy->act(state);
	}

	this->buffer.states.push_back(state);
	this->buffer.actions.push_back(act.action);
	this->buffer.logProbabilities.push_back(act.probability);
	this->buffer.stateValues.push_back(act.critic);

	this->buffer.unrewardedActions++;

	return act.action.detach();
}

uint64_t PPONetwork::predict(double* inputs) {
	auto options = torch::TensorOptions().dtype(torch::kFloat64);
	torch::Tensor inputTensor = torch::from_blob(inputs, { this->inputCount }, options);
	inputTensor = inputTensor.toType(torch::kFloat32);
	// inputTensor = inputTensor.to(torch::Device(torch::kCUDA, 0));

	uint64_t predictionIndex = this->predictions.size();

	this->predictions[predictionIndex] = this->trainAction(inputTensor);

	return predictionIndex;
}

float PPONetwork::getPrediction(uint64_t predictIndex, uint64_t outputIndex) {
	if (this->predictions.find(predictIndex) == this->predictions.end()) {
		return 0.0f; // TODO return a null value
	}

	if (outputIndex >= (uint64_t)this->predictions[predictIndex].size(0)) {
		return 0.0f; // TODO return a null value
	}

	return this->predictions[predictIndex][outputIndex].item<float>();
}

void PPONetwork::finishPredict(uint64_t predictIndex) {
	this->predictions.erase(predictIndex);
}

void PPONetwork::update() {
	// move buffer to GPU
	for (int64_t i = 0; i < this->buffer.rewards.size(); i++) {
		this->buffer.states[i] = this->buffer.states[i].to(torch::Device(torch::kCUDA, 0));
		this->buffer.actions[i] = this->buffer.actions[i].to(torch::Device(torch::kCUDA, 0));
		this->buffer.logProbabilities[i] = this->buffer.logProbabilities[i].to(torch::Device(torch::kCUDA, 0));
		this->buffer.stateValues[i] = this->buffer.stateValues[i].to(torch::Device(torch::kCUDA, 0));
	}

	std::vector<float> rewards;
	float discontinuedReward = 0.0;
	for (int64_t i = this->buffer.rewards.size() - 1; i >= 0; i--) {
		if (this->buffer.isTerminals[i]) {
			discontinuedReward = 0.0;
		}

		discontinuedReward = this->buffer.rewards[i] + (this->gamma * discontinuedReward);
		rewards.insert(rewards.begin(), discontinuedReward);
	}

	this->oldPolicy->setCpuOnly(false);

	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	torch::Tensor rewardsTensor = torch::from_blob(rewards.data(), { (long)rewards.size() }, options);
	rewardsTensor = (rewardsTensor - rewardsTensor.mean()) / (rewardsTensor.std() + 1e-7);
	rewardsTensor = rewardsTensor.to(torch::Device(torch::kCUDA, 0));

	torch::Tensor oldStates = torch::squeeze(torch::stack(this->buffer.states, 0)).detach().to(torch::Device(torch::kCUDA, 0)).reshape({ -1, 1 });
	torch::Tensor oldActions = torch::squeeze(torch::stack(this->buffer.actions, 0)).detach().to(torch::Device(torch::kCUDA, 0));
	torch::Tensor oldLogProbabilities = torch::squeeze(torch::stack(this->buffer.logProbabilities, 0)).detach().to(torch::Device(torch::kCUDA, 0));
	torch::Tensor oldStateValues = torch::squeeze(torch::stack(this->buffer.stateValues, 0)).detach().to(torch::Device(torch::kCUDA, 0));

	torch::Tensor advantages = rewardsTensor.detach() - oldStateValues.detach();

	for (unsigned int i = 0; i < this->epochs; i++) {
		EvaluateResult result = this->policy->evaluate(oldStates, oldActions);

		torch::Tensor stateValues = torch::squeeze(result.critic);

		torch::Tensor ratios = torch::exp(result.probability - oldLogProbabilities.detach());

		torch::Tensor surrogateLoss1 = ratios * advantages;
		torch::Tensor surrogateLoss2 = torch::clamp(ratios, 1 - this->clip, 1 + this->clip) * advantages;

		torch::Tensor actorLoss = -torch::min(surrogateLoss1, surrogateLoss2) + 0.5 * this->loss(stateValues, rewardsTensor) - 0.01 * result.entropy;

		this->actorOptimizer->zero_grad();
		this->criticOptimizer->zero_grad();
		actorLoss.mean().backward();
		this->actorOptimizer->step();
		this->criticOptimizer->step();
	}

	this->oldPolicy.reset();
	this->oldPolicy = std::dynamic_pointer_cast<ActorCritic>(this->policy->clone());
	this->oldPolicy->setCpuOnly(true);

	this->buffer.actions.clear();
	this->buffer.states.clear();
	this->buffer.logProbabilities.clear();
	this->buffer.rewards.clear();
	this->buffer.stateValues.clear();
	this->buffer.isTerminals.clear();
}

void PPONetwork::singleTrain(float reward, bool isTerminal) {
	this->buffer.rewards.push_back(reward);
	this->buffer.isTerminals.push_back(isTerminal);
}

void PPONetwork::train(float reward, bool isTerminal) {
	for (size_t i = 0; i < this->buffer.unrewardedActions; i++) {
		this->buffer.rewards.push_back(reward);
		this->buffer.isTerminals.push_back(isTerminal);
	}

	this->buffer.unrewardedActions = 0;
}

void PPONetwork::setActorLearningRate(float learningRate) {
	this->actorOptimizer = new torch::optim::Adam(
		this->policy->actor.get()->parameters(), torch::optim::AdamOptions(learningRate)
	);
}

void PPONetwork::setCriticLearningRate(float learningRate) {
	this->criticOptimizer = new torch::optim::Adam(
		this->policy->actor.get()->parameters(), torch::optim::AdamOptions(learningRate)
	);
}

void PPONetwork::save(std::string filename) {
	torch::serialize::OutputArchive policyOutput;
	this->policy->save(policyOutput);
	policyOutput.save_to(filename + ".policy");

	torch::serialize::OutputArchive oldPolicyOutput;
	this->oldPolicy->save(oldPolicyOutput);
	oldPolicyOutput.save_to(filename + ".old-policy");
}

void PPONetwork::load(std::string filename) {
	torch::serialize::InputArchive policyInput;
	policyInput.load_from(filename + ".policy");
	this->policy->load(policyInput);

	torch::serialize::InputArchive oldPolicyInput;
	oldPolicyInput.load_from(filename + ".old-policy");
	this->oldPolicy->load(oldPolicyInput);
}
