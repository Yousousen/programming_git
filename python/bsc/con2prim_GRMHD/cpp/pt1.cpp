#include <torch/script.h>
#include <iostream>

int main() {
  // Set the path to the net.pt file.
  char path_to_model[] = "../../net.pt";

  const int N_INPUTS = 14;

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
  // To confirm corresponding output in python, copy over the values of
  // example_input_to_validate_correct_export_and_import and paste it in here.
  float input_data[] = {3.1717e+00,  6.0219e-01,  3.1827e+01,  5.5007e+01,  6.2227e+01,
         -3.8689e+00,  3.3077e+00, -3.5445e+00,  9.3597e-01,  1.6982e-02,
          9.7446e-02,  1.0166e+00,  8.6980e-02,  1.0277e+00};
  auto input_tensor = torch::from_blob(input_data, {1, N_INPUTS});

  // Moving the input tensor to the same device as the model
  input_tensor = input_tensor.to(at::kCPU);

  // Evaluating the model on the input tensor using the forward method
  auto output_tensor = model.forward({input_tensor}).toTensor();

  // Printing the output tensor
  std::cout << "Output: " << output_tensor << "\n";

  return 0;
}
