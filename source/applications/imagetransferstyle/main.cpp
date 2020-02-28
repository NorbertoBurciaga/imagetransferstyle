#include <iostream>
#include <torch/torch.h>

#include "main.h"

using namespace std;

torch::DeviceType IdentifyDeviceType() {
	if (torch::cuda::is_available()) {
		cout << "CUDA available! Training on GPU." << endl;
		return torch::kCUDA;
	} else {
		cout << "Training on CPU." << endl;
		return torch::kCPU;
	}
}

int main(int argc, char* argv[]) {
	torch::manual_seed(1);

	torch::Device device(IdentifyDeviceType());

	return 0;
}
