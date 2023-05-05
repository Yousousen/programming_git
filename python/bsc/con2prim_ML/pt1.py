#!/usr/bin/env python
# coding: utf-8

# # Neural network to learn conservative-to-primitive conversion in relativistic hydrodynamics
# 
# We use Optuna to do a type of Bayesian optimization of the hyperparameters of the model. We then train the model using these hyperparameters to recover the primitive pressure from the conserved variables.
# 
# Use this first cell to convert the notebook to a python script.

# In[2]:


# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import tensorboardX as tbx


# ## Generating the data

# In[3]:


# Checking if GPU is available and setting the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining some constants for convenience
c = 1 # Speed of light
gamma = 5/3 # Adiabatic index

# Defining an analytic equation of state (EOS) for an ideal gas
def eos_analytic(rho, epsilon):
    """Computes the pressure from the rest-mass density and specific internal energy using an analytic EOS.

    Args:
        rho (torch.Tensor): The rest-mass density tensor of shape (batch_size, 1).
        epsilon (torch.Tensor): The specific internal energy tensor of shape (batch_size, 1).

    Returns:
        torch.Tensor: The pressure tensor of shape (batch_size, 1).
    """
    return (gamma - 1) * rho * epsilon


# In[6]:


# Defining a function that generates input data (conserved variables) from random samples of primitive variables
def generate_input_data(n_samples):
    """Generates input data (conserved variables) from random samples of primitive variables.

    Args:
        n_samples (int): The number of samples to generate.

    Returns:
        torch.Tensor: The input data tensor of shape (n_samples, 3).
    """
    # Sampling the primitive variables from uniform distributions
    rho = np.random.uniform(0.1, 10, size=n_samples) # Rest-mass density
    vx = np.random.uniform(-0.9 * c, 0.9 * c, size=n_samples) # Velocity in x-direction
    epsilon = np.random.uniform(0.1, 10, size=n_samples) # Specific internal energy

    # Computing the Lorentz factor from the velocity
    gamma_v = 1 / np.sqrt(1 - vx**2 / c**2)

    # Computing the pressure from the primitive variables using the EOS
    p = eos_analytic(rho, epsilon)

    # Computing the conserved variables from the primitive variables using equations (2) and (3) from the paper 
    D = rho * gamma_v # Conserved density
    h = 1 + epsilon + p / rho # Specific enthalpy
    W = rho * h * gamma_v # Lorentz factor associated with the enthalpy
    Sx = W**2 * vx # Conserved momentum in x-direction
    tau = W**2 - p - D # Conserved energy density

    # Stacking the conserved variables into a numpy array
    x = np.stack([D, Sx, tau], axis=1)

    # Converting the numpy array to a torch tensor and moving it to the device
    x = torch.tensor(x, dtype=torch.float32).to(device)

    # Returning the input data tensor
    return x

# Defining a function that generates output data (labels) from random samples of primitive variables
def generate_output_data(n_samples):
    """Generates output data (labels) from random samples of primitive variables.

    Args:
        n_samples (int): The number of samples to generate.

    Returns:
        torch.Tensor: The output data tensor of shape (n_samples, 1).
    """
    # Sampling the primitive variables from uniform distributions
    rho = np.random.uniform(0.1, 10, size=n_samples) # Rest-mass density
    epsilon = np.random.uniform(0.1, 10, size=n_samples) # Specific internal energy

    # Computing the pressure from the primitive variables using the EOS
    p = eos_analytic(rho, epsilon)

    # Converting the numpy array to a torch tensor and moving it to the device
    y = torch.tensor(p, dtype=torch.float32).to(device)

    # Returning the output data tensor
    return y

# Generating the input and output data for train and test sets using the functions defined
x_train = generate_input_data(8000)
y_train = generate_output_data(8000)
x_test = generate_input_data(2000)
y_test = generate_output_data(2000)

# Checking the shapes of the data tensors
print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test:", y_test.shape)


# ## Defining the neural network class

# In[7]:


# Defining a class for the network
class Net(nn.Module):
    """A class for creating a network with a variable number of hidden layers and units.

    Attributes:
        n_layers (int): The number of hidden layers in the network.
        n_units (list): A list of integers representing the number of units in each hidden layer.
        hidden_activation (torch.nn.Module): The activation function for the hidden layers.
        output_activation (torch.nn.Module): The activation function for the output layer.
        layers (torch.nn.ModuleList): A list of linear layers in the network.
    """

    def __init__(self, n_layers, n_units, hidden_activation, output_activation):
        """Initializes the network with the given hyperparameters.

        Args:
            n_layers (int): The number of hidden layers in the network.
            n_units (list): A list of integers representing the number of units in each hidden layer.
            hidden_activation (torch.nn.Module): The activation function for the hidden layers.
            output_activation (torch.nn.Module): The activation function for the output layer.
        """
        super().__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # Creating a list of linear layers with different numbers of units for each layer
        self.layers = nn.ModuleList([nn.Linear(3, n_units[0])]) # Changed the input size to 3
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(n_units[i - 1], n_units[i]))
        self.layers.append(nn.Linear(n_units[-1], 1)) # Changed the output size to 1

    def forward(self, x):
        """Performs a forward pass on the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 1).
        """
        # Looping over the hidden layers and applying the linear transformation and the activation function
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))

        # Applying the linear transformation and the activation function on the output layer
        x = self.output_activation(self.layers[-1](x))

        # Returning the output tensor
        return x


# ## Setting the search space

# In[8]:


# Defining a function to create a trial network and optimizer
def create_model(trial):
    """Creates a trial network and optimizer based on the sampled hyperparameters.

    Args:
        trial (optuna.trial.Trial): The trial object that contains the hyperparameters.

    Returns:
        tuple: A tuple of (net, loss_fn, optimizer, batch_size, n_epochs,
            scheduler), where net is the trial network,
            loss_fn is the loss function,
            optimizer is the optimizer,
            batch_size is the batch size,
            n_epochs is the number of epochs,
            scheduler is the learning rate scheduler.
    """

    # Sampling the hyperparameters from the search space
    n_layers = trial.suggest_int("n_layers", 1, 5)
    n_units = [trial.suggest_int(f"n_units_{i}", 1, 256) for i in range(n_layers)]
    hidden_activation_name = trial.suggest_categorical("hidden_activation", ["ReLU", "Tanh", "Sigmoid"])
    output_activation_name = trial.suggest_categorical("output_activation", ["Linear", "ReLU"]) # Changed to only linear or ReLU
    loss_name = trial.suggest_categorical("loss", ["MSE", "MAE", "Huber", "LogCosh"]) # Changed to only regression losses
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop", "Adagrad"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1) # Changed the lower bound to 1e-5
    batch_size = trial.suggest_int("batch_size", 1, 512)
    n_epochs = trial.suggest_int("n_epochs", 10, 200)
    scheduler_name = trial.suggest_categorical("scheduler", ["None", "StepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"])

    # Creating the activation functions from their names
    if hidden_activation_name == "ReLU":
        hidden_activation = nn.ReLU()
    elif hidden_activation_name == "Tanh":
        hidden_activation = nn.Tanh()
    else:
        hidden_activation = nn.Sigmoid()

    if output_activation_name == "ReLU":
        output_activation = nn.ReLU()
    else:
        output_activation = nn.Identity() # Changed to use identity function for linear output

    # Creating the loss function from its name
    if loss_name == "MSE":
        loss_fn = nn.MSELoss()
    elif loss_name == "MAE":
        loss_fn = nn.L1Loss()
    elif loss_name == "Huber":
        loss_fn = nn.SmoothL1Loss()
    else:
        # Added creating the log-cosh loss function
        def log_cosh_loss(y_pred, y_true):
            """Computes the log-cosh loss between the predicted and true values.

            Args:
                y_pred (torch.Tensor): The predicted values tensor of shape (batch_size, 1).
                y_true (torch.Tensor): The true values tensor of shape (batch_size, 1).

            Returns:
                torch.Tensor: The log-cosh loss tensor of shape ().
            """
            return torch.mean(torch.log(torch.cosh(y_pred - y_true)))
        
        loss_fn = log_cosh_loss

    # Creating the network with the sampled hyperparameters
    net = Net(n_layers, n_units, hidden_activation, output_activation).to(device) # Added moving the network to the device

    # Creating the optimizer from its name
    if optimizer_name == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=lr)
    else:
        # Added creating the Adagrad optimizer
        optimizer = optim.Adagrad(net.parameters(), lr=lr)

    # Creating the learning rate scheduler from its name
    if scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif scheduler_name == "CosineAnnealingLR":
        # Added creating the CosineAnnealingLR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    elif scheduler_name == "ReduceLROnPlateau":
        # Added creating the ReduceLROnPlateau scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    else:
        scheduler = None

    # Returning the network, the loss function, the optimizer, the batch size, the number of epochs and the scheduler
    return net, loss_fn, optimizer, batch_size, n_epochs, scheduler


# ## The train and eval loop

# In[9]:


# Defining a function to train and evaluate a network on the train and test sets
def train_and_eval(net, loss_fn, optimizer, batch_size, n_epochs, scheduler):
    """Trains and evaluates a network on the train and test sets.

    Args:
        net (Net): The network to train and evaluate.
        loss_fn (torch.nn.Module or function): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        batch_size (int): The batch size to use.
        n_epochs (int): The number of epochs to train for.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): The learning rate scheduler to use.

    Returns:
        tuple: A tuple of (train_losses, test_losses,
            train_metrics, test_metrics), where train_losses and test_losses are lists of average losses per epoch for the train and test sets,
            train_metrics and test_metrics are lists of dictionaries of metrics per epoch for the train and test sets.
    """
    # Creating data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size)

    # Initializing lists to store the losses and metrics for each epoch
    train_losses = []
    test_losses = []
    train_metrics = []
    test_metrics = []

    # Looping over the epochs
    for epoch in range(n_epochs):
        # Setting the network to training mode
        net.train()
        # Initializing variables to store the total loss and metrics for the train set
        train_loss = 0.0
        train_rho_error = 0.0 # Added relative error for rest-mass density
        train_vx_error = 0.0 # Added relative error for velocity in x-direction
        train_epsilon_error = 0.0 # Added relative error for specific internal energy
        # Looping over the batches in the train set
        for x_batch, y_batch in train_loader:
            # Zeroing the gradients
            optimizer.zero_grad()
            # Forward pass
            y_pred = net(x_batch)
            # Reshaping the target tensor to match the input tensor
            y_batch = y_batch.view(-1, 1)
            # Computing the loss
            loss = loss_fn(y_pred, y_batch)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Updating the total loss
            train_loss += loss.item() * x_batch.size(0)
            # Computing the other primitive variables from y_pred and x_batch using equations (A2) to (A5) from the paper 
            D_batch = x_batch[:, 0] # Conserved density
            Sx_batch = x_batch[:, 1] # Conserved momentum in x-direction
            tau_batch = x_batch[:, 2] # Conserved energy density
            rho_pred = D_batch / torch.sqrt(1 + Sx_batch**2 / D_batch**2 / c**2) # Rest-mass density
            vx_pred = Sx_batch / D_batch / c**2 / torch.sqrt(1 + Sx_batch**2 / D_batch**2 / c**2) # Velocity in x-direction
            epsilon_pred = (tau_batch + D_batch) / rho_pred - y_pred / rho_pred - 1 # Specific internal energy
            # Computing the true values of the other primitive variables from y_batch and x_batch using equations (A2) to (A5) from the paper 
            rho_true = D_batch / torch.sqrt(1 + Sx_batch**2 / D_batch**2 / c**2) # Rest-mass density
            vx_true = Sx_batch / D_batch / c**2 / torch.sqrt(1 + Sx_batch**2 / D_batch**2 / c**2) # Velocity in x-direction
            epsilon_true = (tau_batch + D_batch) / rho_true - y_batch / rho_true - 1 # Specific internal energy
            # Updating the total metrics
            # Updating the total metrics
            train_rho_error += torch.mean(torch.abs((rho_pred - rho_true) / rho_true)) * x_batch.size(0) # Relative error for rest-mass density
            train_vx_error += torch.mean(torch.abs((vx_pred - vx_true) / vx_true)) * x_batch.size(0) # Relative error for velocity in x-direction
            train_epsilon_error += torch.mean(torch.abs((epsilon_pred - epsilon_true) / epsilon_true)) * x_batch.size(0) # Relative error for specific internal energy

        # Setting the network to evaluation mode
        net.eval()
        # Initializing variables to store the total loss and metrics for the test set
        test_loss = 0.0
        test_rho_error = 0.0 # Added relative error for rest-mass density
        test_vx_error = 0.0 # Added relative error for velocity in x-direction
        test_epsilon_error = 0.0 # Added relative error for specific internal energy
        # Looping over the batches in the test set
        for x_batch, y_batch in test_loader:
            # Forward pass
            y_pred = net(x_batch)
            # Reshaping the target tensor to match the input tensor
            y_batch = y_batch.view(-1, 1)
            # Computing the loss
            loss = loss_fn(y_pred, y_batch)
            # Updating the total loss
            test_loss += loss.item() * x_batch.size(0)
            # Computing the other primitive variables from y_pred and x_batch using equations (A2) to (A5) from the paper 
            D_batch = x_batch[:, 0] # Conserved density
            Sx_batch = x_batch[:, 1] # Conserved momentum in x-direction
            tau_batch = x_batch[:, 2] # Conserved energy density
            rho_pred = D_batch / torch.sqrt(1 + Sx_batch**2 / D_batch**2 / c**2) # Rest-mass density
            vx_pred = Sx_batch / D_batch / c**2 / torch.sqrt(1 + Sx_batch**2 / D_batch**2 / c**2) # Velocity in x-direction
            epsilon_pred = (tau_batch + D_batch) / rho_pred - y_pred / rho_pred - 1 # Specific internal energy
            # Computing the true values of the other primitive variables from y_batch and x_batch using equations (A2) to (A5) from the paper 
            rho_true = D_batch / torch.sqrt(1 + Sx_batch**2 / D_batch**2 / c**2) # Rest-mass density
            vx_true = Sx_batch / D_batch / c**2 / torch.sqrt(1 + Sx_batch**2 / D_batch**2 / c**2) # Velocity in x-direction
            epsilon_true = (tau_batch + D_batch) / rho_true - y_batch / rho_true - 1 # Specific internal energy
            # Updating the total metrics
            test_rho_error += torch.mean(torch.abs((rho_pred - rho_true) / rho_true)) * x_batch.size(0) # Relative error for rest-mass density
            test_vx_error += torch.mean(torch.abs((vx_pred - vx_true) / vx_true)) * x_batch.size(0) # Relative error for velocity in x-direction
            test_epsilon_error += torch.mean(torch.abs((epsilon_pred - epsilon_true) / epsilon_true)) * x_batch.size(0) # Relative error for specific internal energy

        # Computing the average losses and metrics for the train and test sets
        train_loss /= len(x_train)
        test_loss /= len(x_test)
        train_rho_error /= len(x_train)
        test_rho_error /= len(x_test)
        train_vx_error /= len(x_train)
        test_vx_error /= len(x_test)
        train_epsilon_error /= len(x_train)
        test_epsilon_error /= len(x_test)

        # Appending the losses and metrics to the lists
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_metrics.append({"rho_error": train_rho_error, "vx_error": train_vx_error, "epsilon_error": train_epsilon_error}) # Changed to a dictionary of metrics
        test_metrics.append({"rho_error": test_rho_error, "vx_error": test_vx_error, "epsilon_error": test_epsilon_error}) # Changed to a dictionary of metrics

        # Updating the learning rate scheduler with validation loss if applicable
        if scheduler is not None: 
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # Printing the losses and metrics for the current epoch
        print(f"Epoch {epoch + 1}:")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Train rho error: {train_rho_error:.4f}")
        print(f"Test rho error: {test_rho_error:.4f}")
        print(f"Train vx error: {train_vx_error:.4f}")
        print(f"Test vx error: {test_vx_error:.4f}")
        print(f"Train epsilon error: {train_epsilon_error:.4f}")
        print(f"Test epsilon error: {test_epsilon_error:.4f}")

    # Returning the losses and metrics lists
    return train_losses, test_losses, train_metrics, test_metrics


# ## The objective function

# In[10]:


# Defining a function to be minimized by Optuna
def objective(trial):
    """Creates a trial network and optimizer based on the sampled hyperparameters,
    trains and evaluates it on the train and test sets, and returns the final test loss.

    Args:
        trial (optuna.trial.Trial): The trial object that contains the hyperparameters.

    Returns:
        float: The final test loss.
    """
    # Creating the network and optimizer with the create_model function
    net, loss_fn, optimizer, batch_size, n_epochs, scheduler = create_model(trial)

    # Training and evaluating the network with the train_and_eval function
    train_losses, test_losses, train_metrics, test_metrics = train_and_eval(net, loss_fn, optimizer, batch_size, n_epochs, scheduler)

    # Returning the final test loss
    return test_losses[-1]


# ## Finding the best hyperparameters

# In[11]:


# Creating an Optuna study object
study = optuna.create_study(direction="minimize")

# Running the optimization with a given number of trials
n_trials = 100 # Change this to a larger number for better results
study.optimize(objective, n_trials=n_trials)

# Printing the best hyperparameters and the best value
print("Best hyperparameters:", study.best_params)
print("Best value:", study.best_value)


# ## Training with the best hyperparameters

# In[ ]:


# Creating the network and optimizer with the best hyperparameters
net_best, loss_fn_best, optimizer_best, batch_size_best, n_epochs_best, scheduler_best = create_model(study.best_trial)

# Training and evaluating the network with the best hyperparameters
train_losses_best, test_losses_best, train_metrics_best, test_metrics_best = train_and_eval(net_best, loss_fn_best, optimizer_best, batch_size_best, n_epochs_best, scheduler_best)


# ## Visualizing the results

# In[ ]:


# Plotting the train and test losses per epoch
plt.figure()
plt.plot(range(1, n_epochs_best + 1), train_losses_best, label="Train loss")
plt.plot(range(1, n_epochs_best + 1), test_losses_best, label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curves")
plt.show()

# Plotting the train and test metrics per epoch
plt.figure()
plt.plot(range(1, n_epochs_best + 1), [m["rho_error"] for m in train_metrics_best], label="Train rho error")
plt.plot(range(1, n_epochs_best + 1), [m["rho_error"] for m in test_metrics_best], label="Test rho error")
plt.xlabel("Epoch")
plt.ylabel("Relative error")
plt.legend()
plt.title("Rest-mass density error curves")
plt.show()

plt.figure()
plt.plot(range(1, n_epochs_best + 1), [m["vx_error"] for m in train_metrics_best], label="Train vx error")
plt.plot(range(1, n_epochs_best + 1), [m["vx_error"] for m in test_metrics_best], label="Test vx error")
plt.xlabel("Epoch")
plt.ylabel("Relative error")
plt.legend()
plt.title("Velocity in x-direction error curves")
plt.show()

plt.figure()
plt.plot(range(1, n_epochs_best + 1), [m["epsilon_error"] for m in train_metrics_best], label="Train epsilon error")
plt.plot(range(1, n_epochs_best + 1), [m["epsilon_error"] for m in test_metrics_best], label="Test epsilon error")
plt.xlabel("Epoch")
plt.ylabel("Relative error")
plt.legend()
plt.title("Specific internal energy error curves")
plt.show()


# ## Saving and loading the data and the model

# In[ ]:


# Importing torch and pickle for saving and loading
import torch
import pickle

# Saving the data and the model to files
torch.save(x_train, "x_train.pt")
torch.save(y_train, "y_train.pt")
torch.save(x_test, "x_test.pt")
torch.save(y_test, "y_test.pt")
torch.save(net_best.state_dict(), "net_best.pt")
pickle.dump(study.best_params, open("best_params.pkl", "wb"))

# Loading the data and the model from files
x_train = torch.load("x_train.pt")
y_train = torch.load("y_train.pt")
x_test = torch.load("x_test.pt")
y_test = torch.load("y_test.pt")
net_best = Net(**study.best_params).to(device) # Creating the network with the best hyperparameters
net_best.load_state_dict(torch.load("net_best.pt")) # Loading the network weights
best_params = pickle.load(open("best_params.pkl", "rb")) # Loading the best hyperparameters

