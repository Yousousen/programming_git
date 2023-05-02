#!/usr/bin/env python
# coding: utf-8

# # Experimenting to optimize for h

# In[1]:


# Import the necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Define a new class that inherits from nn.Module
class VariableNetwork(nn.Module):
    # Define the constructor that takes the model as an argument
    def __init__(self, model):
        # Call the parent constructor
        super().__init__()
        # Assign the model to an attribute
        self.model = model
    
    # Override the forward function
    def forward(self, x):
        # Loop over the layers in the ModuleList
        for layer in self.model:
            # Apply the layer to the input
            x = layer(x)
        # Return the final output
        return x


# Define the functions to be approximated
def f(x1, x2, x3):
    return x1 + x2 + x3

def g(x1, x2, x3):
    return x1**2 + x2**3 + 0.5 * x3

def h(x1, x2, x3):
    return x3 * x1**(x2)


# In[2]:


# Define the range and step size for the input variables
x1_range = (0, 10)
x2_range = (0, 10)
x3_range = (0, 10)
dx = 0.5

# Generate the input data by sampling uniformly from the ranges
x1 = np.arange(*x1_range, dx)
x2 = np.arange(*x2_range, dx)
x3 = np.arange(*x3_range, dx)
X1, X2, X3 = np.meshgrid(x1, x2, x3)
X = np.stack([X1.flatten(), X2.flatten(), X3.flatten()], axis=1)

# Compute the output data by applying the functions
Y_f = f(X[:, 0], X[:, 1], X[:, 2])
Y_g = g(X[:, 0], X[:, 1], X[:, 2])
Y_h = h(X[:, 0], X[:, 1], X[:, 2])

# Convert the input and output data to torch tensors
X = torch.from_numpy(X).float()
Y_f = torch.from_numpy(Y_f).float().unsqueeze(1)
Y_g = torch.from_numpy(Y_g).float().unsqueeze(1)
Y_h = torch.from_numpy(Y_h).float().unsqueeze(1)

# Split the data into train and test sets (80% train, 20% test)
train_size = int(0.8 * len(X))
test_size = len(X) - train_size
X_train, X_test = torch.utils.data.random_split(X, [train_size, test_size])
Y_f_train, Y_f_test = torch.utils.data.random_split(Y_f, [train_size, test_size])
Y_g_train, Y_g_test = torch.utils.data.random_split(Y_g, [train_size, test_size])
Y_h_train, Y_h_test = torch.utils.data.random_split(Y_h, [train_size, test_size])


# In[3]:


# Let us have a variable number of hidden layers.
# Define a function to create a neural network with given hyperparameters
def create_network(input_size, output_size, hidden_sizes, activations, output_activation=None):
    # Create a ModuleList to hold the layers
    model = nn.ModuleList()
    # Loop over the hidden sizes and activations
    for hidden_size, activation in zip(hidden_sizes, activations):
        # Add a linear layer with the input size and hidden size
        model.append(nn.Linear(input_size, hidden_size))
        # Use a batch normalization layer between linear and activation layers to improve training stability
        #model.append(nn.BatchNorm1d(hidden_size))
        # Add an activation layer with the given activation function
        model.append(activation())
        # Update the input size for the next layer
        input_size = hidden_size
    # Add the final output layer with the output size
    model.append(nn.Linear(input_size, output_size))
    # If an output activation function is specified, add it to the model
    if output_activation:
        model.append(output_activation())
    # Return the model
    return model



# In[4]:


# Define a function to train a neural network with given hyperparameters and data
def train_network(model, optimizer, loss_fn, batch_size, epochs,
                  X_train, Y_train, X_test=None, Y_test=None):
    # Create a data loader for the training data
    train_loader = DataLoader(
        dataset=torch.utils.data.TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True
    )
    # Initialize a list to store the training losses
    train_losses = []
    # Initialize a list to store the test losses if test data is given
    if X_test is not None and Y_test is not None:
        test_losses = []
    # Loop over the number of epochs
    for epoch in range(epochs):
        # Initialize a variable to store the running loss for this epoch
        running_loss = 0.0
        # Loop over the batches of training data
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass: compute the outputs from the inputs
            outputs = model(inputs)
            # Compute the loss from the outputs and targets
            loss = loss_fn(outputs, targets)
            # Backward pass: compute the gradients from the loss
            loss.backward()
            # Update the parameters using the optimizer
            optimizer.step()
            # Accumulate the running loss
            running_loss += loss.item()
        # Compute and append the average training loss for this epoch
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        # Print the progress
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        # If test data is given, compute and append the test loss for this epoch
        if X_test is not None and Y_test is not None:
            # Compute the outputs from the test inputs
            outputs = model(X_test)
            # Compute the loss from the outputs and test targets
            loss = loss_fn(outputs, Y_test)
            # Append the test loss
            test_loss = loss.item()
            test_losses.append(test_loss)
            # Print the progress
            print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}")
    # Return the train and test losses if test data is given, otherwise return only train losses
    if X_test is not None and Y_test is not None:
        return train_losses, test_losses
    else:
        return train_losses


# In[5]:


# Define a function to plot the losses during training
def plot_losses(train_losses, test_losses=None, function_name=None, hyperparameters=""):
    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(8, 6))
    # Plot the train losses
    ax.plot(train_losses, label="Train Loss")
    # If test losses are given, plot them as well
    if test_losses is not None:
        ax.plot(test_losses, label="Test Loss")
    # Set the title, labels, and legend
    ax.set_title(f"Losses during Training ({hyperparameters})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    # Save and show the plot
    # Use format method to insert hyperparameters into file name
    plt.savefig(f"losses_{function_name}_{hyperparameters}.png")
    plt.show()


# In[6]:


# Define a function to plot the predictions versus the true values
def plot_predictions(model, X, Y_true, function_name, hyperparameters=""):
    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(8, 6))
    # Compute the predictions from the inputs
    Y_pred = model(X).detach().numpy()
    # Plot the predictions and the true values as scatter plots
    ax.scatter(Y_true, Y_pred, label="Predictions", s=2, alpha=0.3)
    ax.scatter(Y_true, Y_true, label="True Values", s=2, alpha=0.3)
    # Set the title, labels, and legend
    ax.set_title(f"Predictions versus True Values ({hyperparameters})")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.legend()
    # Save and show the plot
    # Use format method to insert hyperparameters into file name
    plt.savefig(f"predictions_{function_name}_{hyperparameters}.png")
    plt.show()


# In[7]:


# Define a list of functions to be approximated
functions = [f, g, h]
# Define a list of function names for printing and plotting purposes
function_names = ["f", "g", "h"]
# Define a list of output tensors for each function
outputs = [Y_f, Y_g, Y_h]
# Define a list of output tensors for each function for train and test sets
outputs_train = [Y_f_train, Y_g_train, Y_h_train]
outputs_test = [Y_f_test, Y_g_test, Y_h_test]


# In[8]:


get_ipython().run_cell_magic('script', 'echo skipping', '\n# Loop over each function to be approximated\nfor i in range(len(functions)):\n    # Print the function name\n    print(f"Approximating function {function_names[i]}")\n    # Create a neural network with given hyperparameters\n    input_size = 3 # The number of input variables (x1, x2, x3)\n    output_size = 1 # The number of output variables (y)\n    # Create a network with 3 hidden layers and ReLU activations, and an optional output activation\n    hidden_sizes = [64, 128, 256, 512]\n    activations = [nn.ELU, nn.ELU, nn.ELU, nn.ELU]\n\n\n    output_activation = None\n    model = create_network(input_size, output_size,\n                        hidden_sizes, activations, output_activation=output_activation)\n\n    # Create an instance of VariableNetwork by passing the model\n    network = VariableNetwork(model)\n\n    # Create an optimizer with given hyperparameters\n    optimizer = optim.Adam(network.parameters(), lr=0.001)\n\n    # Create a loss function with given hyperparameters\n    loss_fn = nn.MSELoss()\n    # Train the network with given hyperparameters and data\n    batch_size = 64 # The number of samples in each batch\n    epochs = 100 # The number of times to loop over the whole dataset\n    # Create a string representation of the hyperparameters\n    hyperparameters_str = f"hidden_sizes_{hidden_sizes}_activations_{[act.__name__ for act in activations]}_optimizer_{optimizer.__class__.__name__}_lr_{optimizer.param_groups[0][\'lr\']}_batch_size_{batch_size}_epochs_{epochs}"\n    if output_activation:\n        hyperparameters_str += f"_output_activation_{output_activation.__name__}"\n\n    if output_activation:\n        hyperparameters_str += f"_output_activation_{output_activation.__name__}"\n\n    train_losses, test_losses = train_network(network, optimizer, loss_fn,\n                                            batch_size, epochs,\n                                            X_train.dataset, outputs_train[i].dataset,\n                                            X_test.dataset, outputs_test[i].dataset)\n    plot_losses(train_losses, test_losses, function_names[i], hyperparameters=hyperparameters_str)\n    plot_predictions(network, X, outputs[i], function_names[i], hyperparameters=hyperparameters_str)\n\n    # Save the network with hyperparameters in the file name\n    torch.save(network, f"network_{function_names[i]}_{hyperparameters_str}.pt")\n')


# # Convert notebook to python script
# 
# Running the cell converts this whole notebook to a python script.

# # Ray Tune

# I will now implement Ray Tune to find good parameters for our network.
# I had asked C too to implement more different activation functions, upon which he modified the `create_network` function too.

# In[ ]:


import multiprocessing

num_cpus = multiprocessing.cpu_count()
print(f"Number of CPUs: {num_cpus}")

input_size = 3  # The number of input variables (x1, x2, x3)
output_size = 1  # The number of output variables (y)


# In[ ]:


get_ipython().run_cell_magic('script', 'echo skipping', 'import ray\nfrom ray import tune\nfrom ray.tune.schedulers import ASHAScheduler\nfrom ray.tune.suggest.skopt import SkOptSearch\n\n# Create a function to create a neural network with given hyperparameters\ndef create_network(input_size, output_size, hidden_sizes, activation_classes, output_activation_class=None):\n    # Create a ModuleList to hold the layers\n    model = nn.ModuleList()\n    # Loop over the hidden sizes\n    for hidden_size, activation_class in zip(hidden_sizes, activation_classes):\n        # Add a linear layer with the input size and hidden size\n        model.append(nn.Linear(input_size, hidden_size))\n        # Add an activation layer with the given activation function\n        model.append(activation_class())\n        # Update the input size for the next layer\n        input_size = hidden_size\n    # Add the final output layer with the output size\n    model.append(nn.Linear(input_size, output_size))\n    # If an output activation function is specified, add it to the model\n    if output_activation_class:\n        model.append(output_activation_class())\n    # Return the model\n    return model\n\ndef tune_network(config):\n    activation_classes = [getattr(nn, act_class_name) for act_class_name in config["activation_classes"]]\n    hidden_sizes = config["hidden_sizes"]\n    output_activation_class = getattr(nn, config["output_activation_class"]) if config["output_activation_class"] else None\n\n    model = create_network(input_size, output_size, hidden_sizes, activation_classes, output_activation_class=output_activation_class)\n    network = VariableNetwork(model)\n    optimizer = optim.Adam(network.parameters(), lr=config["lr"])\n    loss_fn = nn.MSELoss()\n\n    train_losses, test_losses = train_network(network, optimizer, loss_fn,\n                                              config["batch_size"], config["epochs"],\n                                              X_train.dataset, Y_f_train.dataset,\n                                              X_test.dataset, Y_f_test.dataset)\n\n    tune.report(test_loss=test_losses[-1])\n')


# In[ ]:


get_ipython().run_cell_magic('script', 'echo skipping', '\nfrom skopt.space import Real, Integer, Categorical\n\n# Define the search space for SkOpt\n# This tries just one hidden layer.\nsearch_space = {\n    "hidden_sizes": Integer(32, 1024),\n    "activation_classes": Categorical(["ReLU", "ELU", "LeakyReLU", "Tanh", "Sigmoid"]),\n    "output_activation_class": Categorical([None, "ReLU", "ELU", "LeakyReLU", "Tanh", "Sigmoid"]),\n    "lr": Real(1e-4, 1e-2, "log-uniform"),\n    "batch_size": Integer(32, 256),\n    "epochs": Integer(10, 50),\n}\n\n# Initialize SkOpt search algorithm\nskopt_search = SkOptSearch(space=search_space, metric="test_loss", mode="min")\n')


# In[ ]:


get_ipython().run_cell_magic('script', 'echo skipping', '\n# Set up the scheduler, searcher, and resources\nasha_scheduler = ASHAScheduler(\n    metric="test_loss",\n    mode="min",\n    max_t=200,\n    grace_period=50,\n    reduction_factor=2\n)\n\nresources_per_trial = {"cpu": num_cpus, "gpu": 0}\n\nfor i in range(len(functions)):\n    # Print the function name\n    print(f"Approximating function {function_names[i]}")\n\n    # Start the tuning process\n    analysis = tune.run(\n        tune_network,\n        search_alg=skopt_search,\n        scheduler=asha_scheduler,\n        num_samples=50,\n        resources_per_trial=resources_per_trial,\n        config=search_space,\n        name=f"tune_network_{function_names[i]}"\n    )\n\n    # Get the best set of hyperparameters\n    best_trial = analysis.get_best_trial("test_loss", "min", "last")\n    best_config = best_trial.config\n    print(f"Best configuration: {best_config}")\n\n    # Train the network with the best hyperparameters\n    best_activation_classes = [getattr(nn, act_class_name) for act_class_name in best_config["activation_classes"]]\n    best_hidden_sizes = best_config["hidden_sizes"]\n    best_output_activation_class = getattr(nn, best_config["output_activation_class"]) if best_config["output_activation_class"] else None\n    best_model = create_network(input_size, output_size, best_hidden_sizes, best_activation_classes, output_activation_class=best_output_activation_class)\n    best_network = VariableNetwork(best_model)\n    best_optimizer = optim.Adam(best_network.parameters(), lr=best_config["lr"])\n    best_loss_fn = nn.MSELoss()\n\n    best_train_losses, best_test_losses = train_network(best_network, best_optimizer, best_loss_fn,\n                                                         best_config["batch_size"], best_config["epochs"],\n                                                         X_train.dataset, outputs_train[i].dataset,\n                                                         X_test.dataset, outputs_test[i].dataset)\n\n    # Print the test loss for the best model\n    print(f"Test loss for the best model: {best_test_losses[-1]}")\n')


# # Optuna
# 
# Let's see if switching to Optuna will avoid having to downgrade numpy.

# In[ ]:


import optuna


# In[ ]:


# Create a function to create a neural network with given hyperparameters
def create_network(input_size, output_size, hidden_sizes, activation_classes, output_activation_class=None):
    # Create a ModuleList to hold the layers
    model = nn.ModuleList()
    # Loop over the hidden sizes
    for hidden_size, activation_class in zip(hidden_sizes, activation_classes):
        # Add a linear layer with the input size and hidden size
        model.append(nn.Linear(input_size, hidden_size))
        # Add an activation layer with the given activation function
        model.append(activation_class())
        # Update the input size for the next layer
        input_size = hidden_size
    # Add the final output layer with the output size
    model.append(nn.Linear(input_size, output_size))
    # If an output activation function is specified, add it to the model
    if output_activation_class:
        model.append(output_activation_class())
    # Return the model
    return model


# In[ ]:


# Modify the function to accept a trial object from Optuna
def tune_network(trial):
    activation_classes = [getattr(nn, trial.suggest_categorical("activation_class", ["ReLU", "ELU", "LeakyReLU", "Tanh", "Sigmoid"])) for _ in range(4)]
    hidden_sizes = [trial.suggest_int("hidden_sizes", 32, 1024) for _ in range(4)]
    output_activation_class_name = trial.suggest_categorical("output_activation_class", [None, "ReLU", "ELU", "LeakyReLU", "Tanh", "Sigmoid"])
    output_activation_class = getattr(nn, output_activation_class_name) if output_activation_class_name else None
    model = create_network(input_size, output_size, hidden_sizes, activation_classes, output_activation_class=output_activation_class)
    network = VariableNetwork(model)
    optimizer = optim.Adam(network.parameters(), lr=trial.suggest_loguniform("lr", 1e-4, 1e-2))
    loss_fn = nn.MSELoss()
    train_losses, test_losses = train_network(network, optimizer, loss_fn,
                                              trial.suggest_int("batch_size", 32, 256), trial.suggest_int("epochs", 10, 50),
                                              X_train.dataset, Y_h_train.dataset, # NOTE: Optimizing FOR h.
                                              X_test.dataset, Y_h_test.dataset) # NOTE: Optimizing FOR h.
    return test_losses[-1]


# Code below optimizes for each function respectively, but I'm currently interested in optimizing for _h_.

# In[ ]:


get_ipython().run_cell_magic('script', 'echo skipping', 'for i in range(len(functions)):\n    # Print the function name\n    print(f"Approximating function {function_names[i]}")\n\n    study = optuna.create_study(direction="minimize")\n    study.optimize(tune_network, n_trials=50)\n    \n    # Get the best set of hyperparameters\n    best_trial = study.best_trial\n    best_config = best_trial.params\n    print(f"Best configuration: {best_config}")\n\n    # Train the network with the best hyperparameters\n    best_activation_classes = [getattr(nn, act_class_name) for act_class_name in best_config["activation_classes"]]\n    best_hidden_sizes = best_config["hidden_sizes"]\n    best_output_activation_class = getattr(nn, best_config["output_activation_class"]) if best_config["output_activation_class"] else None\n    best_model = create_network(input_size, output_size, best_hidden_sizes, best_activation_classes, output_activation_class=best_output_activation_class)\n    best_network = VariableNetwork(best_model)\n    best_optimizer = optim.Adam(best_network.parameters(), lr=best_config["lr"])\n    best_loss_fn = nn.MSELoss()\n\n    best_train_losses, best_test_losses = train_network(best_network, best_optimizer, best_loss_fn,\n                                                         best_config["batch_size"], best_config["epochs"],\n                                                         X_train.dataset, outputs_train[i].dataset,\n                                                         X_test.dataset, outputs_test[i].dataset)\n\n    # Print the test loss for the best model\n    print(f"Test loss for the best model: {best_test_losses[-1]}")\n')


# In[ ]:


# Print the function name
i = 2 # NOTE: function h
print(f"Approximating function {function_names[i]}")

study = optuna.create_study(direction="minimize")
study.optimize(tune_network, n_trials=50)


# In[ ]:


# Get the best set of hyperparameters
best_trial = study.best_trial
best_config = best_trial.params
print(f"Best configuration: {best_config}")


# dict_keys(['activation_class', 'hidden_sizes', 'output_activation_class', 'lr', 'batch_size', 'epochs'])
# Train the network with the best hyperparameters
best_activation_classes = [getattr(nn, act_class_name) for act_class_name in best_config["activation_class"]]
best_hidden_sizes = best_config["hidden_sizes"]
best_output_activation_class = getattr(nn, best_config["output_activation_class"]) if best_config["output_activation_class"] else None


# In[ ]:


best_model = create_network(input_size, output_size, best_hidden_sizes, best_activation_classes, output_activation_class=best_output_activation_class)
best_network = VariableNetwork(best_model)
best_optimizer = optim.Adam(best_network.parameters(), lr=best_config["lr"])
best_loss_fn = nn.MSELoss()

best_train_losses, best_test_losses = train_network(best_network, best_optimizer, best_loss_fn,
                                                        best_config["batch_size"], best_config["epochs"],
                                                        X_train.dataset, outputs_train[i].dataset,
                                                        X_test.dataset, outputs_test[i].dataset)

# Print the test loss for the best model
print(f"Test loss for the best model: {best_test_losses[-1]}")


plot_losses(best_train_losses, best_test_losses, function_names[i])
plot_predictions(best_network, X, outputs[i], function_names[i])

# Save the network with hyperparameters in the file name
torch.save(best_network, f"best_network_{function_names[i]}.pt")


# ## Two hidden layers
