#include "actorCritic.h"

#include "../multivariateNormal.h"

ActorCritic::ActorCritic(unsigned int inputCount, unsigned int outputCount, float initStd, bool cpuOnly) {
	this->inputCount = inputCount;
	this->outputCount = outputCount;

	this->cpuOnly = cpuOnly;

	this->actionVariance = torch::full({ this->outputCount }, initStd * initStd);
	this->std = initStd;

	if (this->cpuOnly) {
		this->actionVariance = this->actionVariance.to(torch::Device(torch::kCPU, 0));
	} else {
		this->actionVariance = this->actionVariance.to(torch::Device(torch::kCUDA, 0));
	}

	this->actor = torch::nn::Sequential(
		torch::nn::Linear(inputCount, 64),
		torch::nn::Tanh(),
		torch::nn::Linear(64, 64),
		torch::nn::Tanh(),
		torch::nn::Linear(64, outputCount)
	);
	register_module("sequential1", this->actor);

	this->critic = torch::nn::Sequential(
		torch::nn::Linear(inputCount, 64),
		torch::nn::Tanh(),
		torch::nn::Linear(64, 64),
		torch::nn::Tanh(),
		torch::nn::Linear(64, 1)
	);
	register_module("sequential2", this->critic);
}

void ActorCritic::setActionStd(float std) {
	this->actionVariance = torch::full({ this->outputCount }, std * std);
	this->std = std;

	if (this->cpuOnly) {
		this->actionVariance = this->actionVariance.to(torch::Device(torch::kCPU, 0));
	} else {
		this->actionVariance = this->actionVariance.to(torch::Device(torch::kCUDA, 0));
	}
}

ActResult ActorCritic::act(torch::Tensor &state) {
	torch::Tensor actionMean = this->actor.get()->forward(state);
	torch::Tensor coVariance = torch::diag(this->actionVariance).unsqueeze(0);

	// get action from normal distribution
	MultivariateNormal distribution = MultivariateNormal(actionMean, coVariance);
	torch::Tensor action = distribution.sample();
	torch::Tensor probability = distribution.logProbability(action);

	return ActResult {
		action: action.detach(),
		probability: probability.detach(),
		critic: this->critic.get()->forward(state).detach(),
	};
}

EvaluateResult ActorCritic::evaluate(torch::Tensor &state, torch::Tensor &action) {
	torch::Tensor actionMean = this->actor.get()->forward(state);

	torch::Tensor actionVariance = this->actionVariance.expand_as(actionMean);
	torch::Tensor coVariance = torch::diag_embed(actionVariance);

	if (this->cpuOnly) {
		coVariance = coVariance.to(torch::Device(torch::kCPU, 0));
	} else {
		coVariance = coVariance.to(torch::Device(torch::kCUDA, 0));
	}

	MultivariateNormal distribution = MultivariateNormal(actionMean, coVariance);

	if (this->outputCount == 1) {
		action = action.reshape({ -1, this->outputCount });
	}

	return EvaluateResult {
		probability: distribution.logProbability(action), // no detach
		critic: this->critic.get()->forward(state),
		entropy: distribution.entropy(),
	};
}

void ActorCritic::reset() {
	this->actor = torch::nn::Sequential(
		torch::nn::Linear(inputCount, 64),
		torch::nn::Linear(64, 64),
		torch::nn::Linear(64, outputCount)
	);
	register_module("sequential1", this->actor);

	if (this->cpuOnly) {
		this->actor.get()->to(torch::Device(torch::kCPU, 0));
	} else {
		this->actor.get()->to(torch::Device(torch::kCUDA, 0));
	}

	this->critic = torch::nn::Sequential(
		torch::nn::Linear(inputCount, 64),
		torch::nn::Linear(64, 64),
		torch::nn::Linear(64, 1)
	);
	register_module("sequential2", this->critic);

	if (this->cpuOnly) {
		this->critic.get()->to(torch::Device(torch::kCPU, 0));
	} else {
		this->critic.get()->to(torch::Device(torch::kCUDA, 0));
	}
}

void ActorCritic::copy(ActorCritic &source) {
	torch::autograd::GradMode::set_enabled(false);

	auto new_params = source.named_parameters();
	auto params = this->named_parameters(true);
	auto buffers = this->named_buffers(true);
	for (auto& val : new_params) {
		auto name = val.key();
		auto* t = params.find(name);
		if (t != nullptr) {
			t->copy_(val.value());
		} else {
			t = buffers.find(name);
			if (t != nullptr) {
				t->copy_(val.value());
			}
		}
	}

	torch::autograd::GradMode::set_enabled(true);
}

void ActorCritic::setCpuOnly(bool cpuOnly) {
	this->cpuOnly = cpuOnly;

	this->setActionStd(this->std);

	if (cpuOnly) {
		this->to(torch::Device(torch::kCPU, 0));
		this->actor.get()->to(torch::Device(torch::kCPU, 0));
		this->critic.get()->to(torch::Device(torch::kCPU, 0));
	} else {
		this->to(torch::Device(torch::kCUDA, 0));
		this->actor.get()->to(torch::Device(torch::kCUDA, 0));
		this->critic.get()->to(torch::Device(torch::kCUDA, 0));
	}
}
