#include <fstream>
#include <iostream>
#include <torch/torch.h>

#include "gnuplot-iostream.h"

#include "api/include.h"

#include "dqnNetwork.h"
#include "runtime.h"

#include "ppo/ppoNetwork.h"

int main() {
	// torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kFloat64));

	float currentStd = 0.01;
	float actorLearningRate = 0.0003 / 5.0;
	float criticLearningRate = 0.001 / 5.0;
	PPONetwork test(1, 1, actorLearningRate, criticLearningRate, 0.999, 50, 0.2, currentStd);
	test.load("model.dat");

	unsigned int iterations = 0;
	int64_t times = 1000;

	Gnuplot gp;

	for (unsigned int i = 0; i < 500; i++) {
		float rewardAcc = 0.0;
		float averageDistance = 0.0;

		for (unsigned int j = 0; j < times; j++) {
			float inputNumber = M_PI * ((float)j / (float)times);
			// if (i % 2 == 0) {
			// 	inputNumber = M_PI * ((float)(times - j) / (float)times);
			// }

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

		if (i % 100 == 0 && i != 0) {
			currentStd = test.decayActionStd(0.005, 0.001);

			actorLearningRate /= 10.0;
			test.setActorLearningRate(actorLearningRate);

			criticLearningRate /= 10.0;
			test.setCriticLearningRate(criticLearningRate);

			test.save("model2.dat");
		}

		test.update();

		std::cout << rewardAcc << " " << (rewardAcc / (float)times) << " " << currentStd << std::endl;

		std::vector<std::pair<float, float>> graphData;
		std::vector<std::pair<float, float>> truthData;
		for (unsigned int i = 0; i < times; i++) {
			float inputNumber = M_PI * ((float)i / (float)times);
			torch::Tensor input = torch::full({ 1 }, inputNumber).to(torch::Device(torch::kCUDA, 0));
			torch::Tensor output = test.predict(input).to(torch::Device(torch::kCUDA, 0));

			graphData.push_back(std::make_pair(inputNumber, output.item<float>()));
			truthData.push_back(std::make_pair(inputNumber, (float)sin(inputNumber)));
		}

		gp << "set xrange [0:3.14159]\n";
		gp << "set yrange [-0.5:1.5]\n";
		gp << "plot '-' with lines title 'prediction', '-' with lines title 'truth'\n";
		gp.send1d(graphData);
		gp.send1d(truthData);
	}

	std::ofstream file("graph");
	for (unsigned int i = 0; i < times; i++) {
		float inputNumber = M_PI * ((float)i / (float)times);
		torch::Tensor input = torch::full({ 1 }, inputNumber).to(torch::Device(torch::kCUDA, 0));
		torch::Tensor output = test.predict(input).to(torch::Device(torch::kCUDA, 0));

		file << inputNumber << "," << output.item<float>() << std::endl;
	}
}
