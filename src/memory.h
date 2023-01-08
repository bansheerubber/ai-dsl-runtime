#include <torch/torch.h>
#include <vector>

struct Transition {
	torch::Tensor action;
	torch::Tensor nextState;
	torch::Tensor reward;
	torch::Tensor state;
};

class Memory {
public:
	void push(Transition transition);
	std::vector<Transition> sample(unsigned int batchSize);

private:
	std::vector<Transition> memory;
};