#include "ppoNetwork.h"

PPONetwork::PPONetwork(
	unsigned int inputCount,
	unsigned int outputCount,
	double actorLearningRate,
	double criticLearningRate,
	double gamma,
	unsigned int epochs,
	double clip,
	double actionStd
) {
	this->gamma = gamma;
	this->clip = clip;
	this->epochs = epochs;

	this->actionStd = actionStd;

	this->policy = std::make_shared<ActorCritic>(inputCount, outputCount, actionStd);
	this->policy->to(torch::Device(torch::kCUDA, 0));

	this->actorOptimizer = new torch::optim::Adam(
		this->policy->actor.get()->parameters(), torch::optim::AdamOptions(actorLearningRate)
	);

	this->criticOptimizer = new torch::optim::Adam(
		this->policy->critic.get()->parameters(), torch::optim::AdamOptions(criticLearningRate)
	);

	this->oldPolicy = std::dynamic_pointer_cast<ActorCritic>(this->policy->clone());
	this->oldPolicy->to(torch::Device(torch::kCUDA, 0));
}

void PPONetwork::setActionStd(double std) {
	this->actionStd = std;
	this->policy->setActionStd(std);
	this->oldPolicy->setActionStd(std);
}

void PPONetwork::decayActionStd(double decayRate, double minimum) {
	this->actionStd = this->actionStd - decayRate;
	// TODO round action STD

	if (this->actionStd < minimum) {
		this->actionStd = minimum;
	}

	this->setActionStd(this->actionStd);
}

torch::Tensor PPONetwork::selectAction(torch::Tensor &state) {
	ActResult act;
	{
		torch::NoGradGuard no_grad;
		act = this->oldPolicy->act(state);
	}

	this->buffer.states.push_back(state);
	this->buffer.actions.push_back(act.action);
	this->buffer.logProbabilities.push_back(act.probability);
	this->buffer.stateValues.push_back(act.critic);

	return act.action.detach();
}

void PPONetwork::update() {
	std::vector<double> rewards;
	double discontinuedReward = 0.0;
	for (int64_t i = this->buffer.rewards.size() - 1; i >= 0; i--) {
		if (this->buffer.isTerminals[i]) {
			discontinuedReward = 0.0;
		}

		discontinuedReward = this->buffer.rewards[i] + (this->gamma * discontinuedReward);
		rewards.insert(rewards.begin(), discontinuedReward);
	}

	auto options = torch::TensorOptions().dtype(torch::kFloat64);
	torch::Tensor rewardsTensor = torch::from_blob(rewards.data(), { (long)rewards.size() }, options).to(torch::kFloat32);
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

	this->buffer.actions.clear();
	this->buffer.states.clear();
	this->buffer.logProbabilities.clear();
	this->buffer.rewards.clear();
	this->buffer.stateValues.clear();
	this->buffer.isTerminals.clear();
}

void PPONetwork::singleTrain(double reward, bool isTerminal) {
	this->buffer.rewards.push_back(reward);
	this->buffer.isTerminals.push_back(isTerminal);
}

// use the current buffer state for backpropagation
void PPONetwork::train(double reward, bool isTerminal) {
	// TODO optimize this
	for (size_t i = 0; i < this->buffer.actions.size(); i++) {
		this->buffer.rewards.push_back(reward);
		this->buffer.isTerminals.push_back(isTerminal);
	}

	// TODO decay action STD

	this->update();
}
