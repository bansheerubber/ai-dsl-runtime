#include <fstream>
#include <iostream>
#include <torch/torch.h>

#include "api/include.h"

#include "dqnNetwork.h"
#include "runtime.h"

#include "ppo/ppoNetwork.h"

int main() {
	double currentStd = 0.2;
	PPONetwork test(1, 1, 0.0003, 0.001, 0.999, 50, 0.2, currentStd);

	unsigned int iterations = 0;
	int64_t times = 1000;

	for (unsigned int i = 0; i < 500; i++) {
		float rewardAcc = 0.0;
		float averageDistance = 0.0;

		for (unsigned int j = 0; j < times; j++) {
			float inputNumber = M_PI * ((float)j / (float)times);

			torch::Tensor input = torch::full({ 1 }, inputNumber).to(torch::Device(torch::kCUDA, 0));
			torch::Tensor expected = torch::sin(input).to(torch::Device(torch::kCUDA, 0));
			torch::Tensor output = test.selectAction(input).to(torch::Device(torch::kCUDA, 0));

			float distance = torch::abs(output - expected).item<float>();
			float reward = pow(20.0, -5.0 * (distance - 0.2));

			rewardAcc += reward;
			averageDistance += distance / (float)times;
			test.singleTrain(reward, false);

			iterations++;
		}

		if (i % 100 == 0) {
			currentStd = test.decayActionStd(0.05, 0.03);
		}

		std::cout << rewardAcc << " " << (rewardAcc / (float)times) << " " << currentStd << std::endl;

		test.update();
	}

	std::ofstream file("graph");
	for (unsigned int i = 0; i < times; i++) {
		float inputNumber = M_PI * ((float)i / (float)times);
		torch::Tensor input = torch::full({ 1 }, inputNumber).to(torch::Device(torch::kCUDA, 0));
		torch::Tensor output = test.selectAction(input).to(torch::Device(torch::kCUDA, 0));

		file << inputNumber << "," << output.item<float>() << std::endl;
	}
}
