#include <torch/script.h>

// Checking if CUDA is available
bool cuda_available = torch::cuda::is_available();

// Setting the device accordingly
at::Device device = cuda_available ? at::kCUDA : at::kCPU;

// Loading the model from the file
std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("../model.pt");

// Converting the input data to a tensor
torch::Tensor input = torch::tensor({{ 4.9706,  0.3188,  8.5965},
                                     { 7.9518,  5.3166,  4.3170},
                                     { 5.8308, 14.7490, 14.6609},
                                     { 4.6200,  0.0829,  2.7031},
                                     { 1.5898,  3.1772,  3.5144},
                                     { 0.1266,  0.1761,  0.1656},
                                     { 8.0950, 10.3077, 15.3977},
                                     { 4.2992,  8.2627,  6.6898},
                                     {11.2275, 10.1282,  7.7118},
                                     { 2.2244,  0.0301,  3.4477},
                                     { 3.6509,  2.7749,  2.8721},
                                     { 6.7411, 18.4558, 17.8222},
                                     { 9.5191, 24.9498, 24.5091},
                                     { 5.3015,  4.9007,  3.7724},
                                     { 1.4423,  1.3792,  1.0336},
                                     { 3.4728,  2.6612,  2.0619},
                                     { 5.1982,  1.5509,  1.6710},
                                     { 7.4406,   .4610 ,   .5647 },
                                     {   .8908 ,   .3629 ,   .9034 },
                                     {   .3611 ,   .4911 ,   .3751 }});


// Specifying the device for the module and the input tensor
module->to(device);
input = input.to(device);

// Evaluating the model on the input tensor
torch::Tensor output = module->forward({input}).toTensor();

