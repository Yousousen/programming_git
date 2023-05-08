#include <torch/script.h>
#include <iostream>

int main() {
  // Declaring a variable to store the model
  torch::jit::script::Module model;

  // Loading the model from the file "net.pt" using the torch::jit::load function
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load("net.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  // Printing a message to indicate successful loading
  std::cout << "Model loaded successfully\n";

  // Checking if CUDA is available
  //bool cuda_available = torch::cuda::is_available();

  // Setting the device accordingly
  //at::Device device = cuda_available ? at::kCUDA : at::kCPU;
  at::Device device = at::kCPU;


  // Converting the input data from python to C++ using torch::from_blob function
  float input_data[] = {12.0043, 32.6378, 31.5031};
  auto input_tensor = torch::from_blob(input_data, {1, 3});

  // Moving the input tensor to the same device as the model
  input_tensor = input_tensor.to(model.device());

  // Evaluating the model on the input tensor using the forward method
  auto output_tensor = model.forward({input_tensor}).toTensor();

  // Printing the output tensor
  std::cout << "Output: " << output_tensor << "\n";

  return 0;
}