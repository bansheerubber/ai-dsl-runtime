#pragma once

#include <cmath>
#include <torch/torch.h>

// class ported from pytorch python code, needs a 1:1 port or else computational graph will be messed up and the PPO
// will not learn
class MultivariateNormal {
public:
	MultivariateNormal(torch::Tensor &mean, torch::Tensor &covariance);

	torch::Tensor sample();
	torch::Tensor logProbability(torch::Tensor &tensor);
	torch::Tensor entropy();

private:
	torch::Tensor mean;
	torch::Tensor covariance;
	torch::Tensor choleskyCovariance;

	std::vector<int64_t> batchShape;
	int64_t eventShape;
};
