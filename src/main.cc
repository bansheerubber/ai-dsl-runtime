#include <torch/torch.h>
#include <iostream>

#include "api/include.h"

#include "dqnNetwork.h"
#include "runtime.h"

#include "ppo/ppoNetwork.h"

int main() {
	PPONetwork test(1, 1, 0.0003, 0.001, 0.99, 80, 0.2, 0.1);

	unsigned int iterations = 0;

	while (true) {
		double rewardAcc = 0.0;
		for (unsigned int i = 0; i < 1000; i++) {
			torch::Tensor input = torch::randn({ 1 }).to(torch::Device(torch::kCUDA, 0));
			torch::Tensor expected = torch::sin(input).to(torch::Device(torch::kCUDA, 0));
			torch::Tensor output = test.selectAction(input).to(torch::Device(torch::kCUDA, 0));

			double distance = torch::abs(output - expected).item<double>();
			double reward = pow(20.0, -5.0 * (distance - 0.2));

			rewardAcc += reward;
			test.singleTrain(reward, false);

			iterations++;
		}

		std::cout << rewardAcc << " " << (rewardAcc / 1000.0) << std::endl;

		test.update();
	}

	// ActorCritic test(2, 2, 0.1);
	// test.to(torch::Device(torch::kCUDA, 0));

	// torch::Tensor state = torch::randn({ 100, 2 }).to(torch::Device(torch::kCUDA, 0));
	// torch::Tensor action = torch::randn({ 100, 2 }).to(torch::Device(torch::kCUDA, 0));

	// test.evaluate(state, action);
}
