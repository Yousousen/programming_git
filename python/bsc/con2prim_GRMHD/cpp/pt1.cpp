#include <torch/script.h>
#include <iostream>

int main() {
  // Set the path to the net.pt file.
  char path_to_model[] = "../../net.pt";

  // Declaring a variable to store the model
  torch::jit::script::Module model;

  // Loading the model from the file "net.pt" using the torch::jit::load function
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = torch::jit::load(path_to_model);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model, did you correctly set the path to the net.pt file?\n";
    return -1;
  }

  // Printing a message to indicate successful loading
  std::cout << "Model loaded successfully\n";

  // Checking if CUDA is available
  //bool cuda_available = torch::cuda::is_available();

  // Setting the device accordingly
  //at::Device device = cuda_available ? at::kCUDA : at::kCPU;
  //at::Device device = at::kCPU;

  // Moving the model to the same device as the input tensor
  model.to(at::kCPU);


  // Converting the input data from python to C++ using torch::from_blob function
  float input_data[] = {7.7606, 3.8938, 5.0599, 4.8451, 9.1032, 0.2814, 0.7417, 0.5824};
  auto input_tensor = torch::from_blob(input_data, {1, 8});

  // Moving the input tensor to the same device as the model
  input_tensor = input_tensor.to(at::kCPU);

  // Evaluating the model on the input tensor using the forward method
  auto output_tensor = model.forward({input_tensor}).toTensor();

  // Printing the output tensor
  std::cout << "Output: " << output_tensor << "\n";

  return 0;
}
