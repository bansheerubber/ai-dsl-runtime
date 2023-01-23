#include "multivariateNormal.h"

#include <algorithm>
#include <tuple>

std::vector<int64_t> broadcastShapes(torch::Tensor &mean, torch::Tensor &covariance) {
	int64_t maxLength = covariance.dim() - 2 > mean.dim() - 1 ? covariance.dim() - 2 : mean.dim() - 1;

	std::vector<int64_t> shape;
	shape.resize(maxLength);

	for (int64_t i = 0; i < maxLength; i++) {
		shape[i] = 1;
	}

	// broadcast for covariance
	for (int64_t i = covariance.dim() - 3; i >= 0; i--) {
		if (covariance.size(i) == 1 || covariance.size(i) == shape[i]) {
			continue;
		}

		shape[i] = covariance.size(i);
	}

	// broadcast for mean
	for (int64_t i = mean.dim() - 2; i >= 0; i--) {
		if (mean.size(i) == 1 || mean.size(i) == shape[i]) {
			continue;
		}

		shape[i] = mean.size(i);
	}

	return shape;
}

torch::Tensor evilDistance(torch::Tensor bL, torch::Tensor bx) {
	int64_t n = bx.size(-1);
	std::vector<int64_t> bxBatchShape;
	for (int64_t i = 0; i < bx.dim() - 1; i++) {
		bxBatchShape.push_back(bx.size(i));
	}

	int64_t bxBatchDims = bxBatchShape.size();
	int64_t bLBatchDims = bL.dim() - 2;

	int64_t outerBatchDims = bxBatchDims - bLBatchDims;
	int64_t oldBatchDims = outerBatchDims + bLBatchDims;
	int64_t newBatchDims = outerBatchDims + 2 * bLBatchDims;

	std::vector<int64_t> bxNewShape;
	for (int64_t i = 0; i < outerBatchDims; i++) {
		bxNewShape.push_back(bx.size(i));
	}

	for (int64_t i = 0; i < bL.dim() - 2 && i < bx.dim() - outerBatchDims - 1; i++) {
		bxNewShape.push_back(bx.size(i) / bL.size(i));
		bxNewShape.push_back(bL.size(i));
	}
	bxNewShape.push_back(n);

	bx = bx.reshape(at::IntArrayRef(bxNewShape));

	std::vector<int64_t> permuteDims;
	for (int64_t i = 0; i < outerBatchDims; i++) {
		permuteDims.push_back(i);
	}

	for (int64_t i = outerBatchDims; i < newBatchDims; i += 2) {
		permuteDims.push_back(i);
	}

	for (int64_t i = outerBatchDims + 1; i < newBatchDims; i += 2) {
		permuteDims.push_back(i);
	}

	permuteDims.push_back(newBatchDims);

	bx = bx.permute(permuteDims);

	torch::Tensor flatL = bL.reshape({ -1, n, n });
	torch::Tensor flatX = bx.reshape({ -1, flatL.size(0), n });
	torch::Tensor flatXSwap = flatX.permute({ 1, 2, 0 });
	torch::Tensor mSwap = torch::linalg::solve_triangular(flatL, flatXSwap, false, true, false).pow(2).sum(-2);
	torch::Tensor M = mSwap.t();

	std::vector<int64_t> bxShape;
	for (int64_t i = 0; i < bx.dim() - 1; i++) {
		bxShape.push_back(bx.size(i));
	}
	torch::Tensor permutedM = M.reshape(at::IntArrayRef(bxShape));

	std::vector<int64_t> permuteInverseDims;
	for (int64_t i = 0; i < outerBatchDims; i++) {
		permuteInverseDims.push_back(i);
	}

	for (int64_t i = 0; i < bLBatchDims; i++) {
		permuteInverseDims.push_back(outerBatchDims + i);
		permuteInverseDims.push_back(oldBatchDims + i);
	}

	torch::Tensor reshapedM = permutedM.permute(permuteInverseDims);
	return reshapedM.reshape(bxBatchShape);
}

MultivariateNormal::MultivariateNormal(torch::Tensor &mean, torch::Tensor &covariance) {
	this->batchShape = broadcastShapes(mean, covariance);

	std::vector<int64_t> covarianceShape(this->batchShape);
	covarianceShape.push_back(-1);
	covarianceShape.push_back(-1);

	std::vector<int64_t> meanShape(this->batchShape);
	meanShape.push_back(-1);

	covariance.expand(at::IntArrayRef(covarianceShape));
	mean.expand(at::IntArrayRef(meanShape));

	this->mean = mean;
	this->covariance = covariance;

	this->eventShape = mean.size(mean.dim() - 1);
	this->choleskyCovariance = torch::linalg::cholesky(covariance);
}

torch::Tensor MultivariateNormal::sample() {
	std::vector<int64_t> shape(this->batchShape);
	shape.push_back(this->eventShape);

	torch::Tensor eps = torch::empty(at::IntArrayRef(shape)).normal_().to(this->mean.device());
	return this->mean + torch::matmul(this->choleskyCovariance, eps.unsqueeze(-1)).squeeze(-1);
}

torch::Tensor MultivariateNormal::logProbability(torch::Tensor &tensor) {
	torch::Tensor delta = tensor - this->mean;
	torch::Tensor M = evilDistance(this->choleskyCovariance, delta);
	auto halfLogDeterminant = this->choleskyCovariance.diagonal(0, -2, -1).log().sum(-1);
	return -0.5 * (this->eventShape * log(2.0 * M_PI) + M) - halfLogDeterminant;
}

torch::Tensor MultivariateNormal::entropy() {
	auto halfLogDeterminant = this->choleskyCovariance.diagonal(0, -2, -1).log().sum(-1);
	torch::Tensor entropy = 0.5 * this->eventShape * (1.0 + log(2.0 * M_PI)) + halfLogDeterminant;
	if (this->batchShape.size() == 0) {
		return entropy;
	} else {
		return entropy.expand(at::IntArrayRef(this->batchShape));
	}
}
