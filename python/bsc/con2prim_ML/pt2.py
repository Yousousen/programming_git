#!/usr/bin/env python
# coding: utf-8

# # Neural network to learn conservative-to-primitive conversion in relativistic hydrodynamics
# 
# We use Optuna to do a type of Bayesian optimization of the hyperparameters of the model. We then train the model using these hyperparameters to recover the primitive pressure from the conserved variables.
# 
# Use this first cell to convert this notebook to a python script.

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
c = 1  # Speed of light
gamma = 5 / 3  # Adiabatic index


# Defining an analytic equation of state (EOS) for an ideal gas
def eos_analytic(rho, epsilon):
    """Computes the pressure from rest-mass density and specific internal energy using an analytic EOS.

    Args:
        rho (torch.Tensor): The rest-mass density tensor of shape (n_samples,).
        epsilon (torch.Tensor): The specific internal energy tensor of shape (n_samples,).

    Returns:
        torch.Tensor: The pressure tensor of shape (n_samples,).
    """
    return (gamma - 1) * rho * epsilon


# In[4]:


# Defining a function that samples primitive variables from uniform distributions
def sample_primitive_variables(n_samples):
    """Samples primitive variables from uniform distributions.

    Args:
        n_samples (int): The number of samples to generate.

    Returns:
        tuple: A tuple of (rho, vx, epsilon), where rho is rest-mass density,
            vx is velocity in x-direction,
            epsilon is specific internal energy,
            each being a numpy array of shape (n_samples,).
    """
    # Sampling from uniform distributions with intervals matching Dieseldorst et al.
    rho = np.random.uniform(0, 10.1, size=n_samples)  # Rest-mass density
    vx = np.random.uniform(0, 0.721 * c, size=n_samples)  # Velocity in x-direction
    epsilon = np.random.uniform(0, 2.02, size=n_samples)  # Specific internal energy

    # Returning the primitive variables
    return rho, vx, epsilon


# Defining a function that computes conserved variables from primitive variables
def compute_conserved_variables(rho, vx, epsilon):
    """Computes conserved variables from primitive variables using equations (2) and (3) from the paper.

    Args:
        rho (torch.Tensor): The rest-mass density tensor of shape (n_samples,).
        vx (torch.Tensor): The velocity in x-direction tensor of shape (n_samples,).
        epsilon (torch.Tensor): The specific internal energy tensor of shape (n_samples,).

    Returns:
        tuple: A tuple of (D, Sx, tau), where D is conserved density,
            Sx is conserved momentum in x-direction,
            tau is conserved energy density,
            each being a torch tensor of shape (n_samples,).
    """
    # Computing the Lorentz factor from the velocity
    gamma_v = 1 / torch.sqrt(1 - vx ** 2 / c ** 2)

    # Computing the pressure from the primitive variables using the EOS
    p = eos_analytic(rho, epsilon)

    # Computing the conserved variables from the primitive variables
    D = rho * gamma_v  # Conserved density
    h = 1 + epsilon + p / rho  # Specific enthalpy
    W = rho * h * gamma_v  # Lorentz factor associated with the enthalpy
    Sx = W ** 2 * vx  # Conserved momentum in x-direction
    tau = W ** 2 - p - D  # Conserved energy density

    # Returning the conserved variables
    return D, Sx, tau


# Defining a function that generates input data (conserved variables) from random samples of primitive variables
def generate_input_data(n_samples):
    """Generates input data (conserved variables) from random samples of primitive variables.

    Args:
        n_samples (int): The number of samples to generate.

    Returns:
        torch.Tensor: The input data tensor of shape (n_samples, 3).
    """
    # Sampling the primitive variables using the sample_primitive_variables function
    rho, vx, epsilon = sample_primitive_variables(n_samples)

    # Converting the numpy arrays to torch tensors and moving them to the device
    rho = torch.tensor(rho, dtype=torch.float32).to(device)
    vx = torch.tensor(vx, dtype=torch.float32).to(device)
    epsilon = torch.tensor(epsilon, dtype=torch.float32).to(device)

    # Computing the conserved variables using the compute_conserved_variables function
    D, Sx, tau = compute_conserved_variables(rho, vx, epsilon)

    # Stacking the conserved variables into a torch tensor
    x = torch.stack([D, Sx, tau], axis=1)

    # Returning the input data tensor
    return x

# Defining a function that generates output data (labels) from random samples of primitive variables
def generate_labels(n_samples):
    """Generates output data (labels) from random samples of primitive variables.

    Args:
        n_samples (int): The number of samples to generate.

    Returns:
        torch.Tensor: The output data tensor of shape (n_samples,).
    """
    # Sampling the primitive variables using the sample_primitive_variables function
    rho, _, epsilon = sample_primitive_variables(n_samples)

    # Converting the numpy arrays to torch tensors and moving them to the device
    rho = torch.tensor(rho, dtype=torch.float32).to(device)
    epsilon = torch.tensor(epsilon, dtype=torch.float32).to(device)

    # Computing the pressure from the primitive variables using the EOS
    p = eos_analytic(rho, epsilon)

    # Returning the output data tensor
    return p


# In[5]:


# Generating the input and output data for train and test sets using the functions defined
# Using the same number of samples as Dieseldorst et al.
x_train = generate_input_data(80000)
y_train = generate_labels(80000)
x_test = generate_input_data(10000)
y_test = generate_labels(10000)

# Checking the shapes of the data tensors
print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test:", y_test.shape)


# ## Defining the neural network

# In[6]:


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
        self.layers = nn.ModuleList([nn.Linear(3, n_units[0])])  # Changed the input size to 3
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(n_units[i - 1], n_units[i]))
        self.layers.append(nn.Linear(n_units[-1], 1))  # Changed the output size to 1

        # Adding some assertions to check that the input arguments are valid
        assert isinstance(n_layers, int) and n_layers > 0, "n_layers must be a positive integer"
        assert isinstance(n_units, list) and len(n_units) == n_layers, "n_units must be a list of length n_layers"
        assert all(isinstance(n, int) and n > 0 for n in n_units), "n_units must contain positive integers"
        assert isinstance(hidden_activation, nn.Module), "hidden_activation must be a torch.nn.Module"
        assert isinstance(output_activation, nn.Module), "output_activation must be a torch.nn.Module"

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


# ## Defining the model and search space

# In[7]:


# Defining a function to create a trial network and optimizer
def create_model(trial):
    """Creates a trial network and optimizer based on the sampled hyperparameters.

    Args:
        trial (optuna.trial.Trial): The trial object that contains the hyperparameters.

    Returns:
        tuple: A tuple of (net, loss_fn, optimizer, batch_size, n_epochs,
            scheduler, loss_name, optimizer_name, scheduler_name,
            n_units, n_layers, hidden_activation, output_activation),
            where net is the trial network,
            loss_fn is the loss function,
            optimizer is the optimizer,
            batch_size is the batch size,
            n_epochs is the number of epochs,
            scheduler is the learning rate scheduler,
            loss_name is the name of the loss function,
            optimizer_name is the name of the optimizer,
            scheduler_name is the name of the scheduler,
            n_units is a list of integers representing the number of units in each hidden layer,
            n_layers is an integer representing the number of hidden layers in the network,
            hidden_activation is a torch.nn.Module representing the activation function for the hidden layers,
            output_activation is a torch.nn.Module representing the activation function for the output layer.
    """

    # Sampling the hyperparameters from the search space
    n_layers = trial.suggest_int("n_layers", 1, 5)
    n_units = [trial.suggest_int(f"n_units_{i}", 1, 256) for i in range(n_layers)]
    hidden_activation_name = trial.suggest_categorical(
        "hidden_activation", ["ReLU", "Tanh", "Sigmoid"]
    )
    output_activation_name = trial.suggest_categorical(
        "output_activation", ["Linear", "ReLU"]
    )  # Changed to only linear or ReLU
    loss_name = trial.suggest_categorical(
        "loss", ["MSE", "MAE", "Huber", "LogCosh"]
    )  # Changed to only regression losses
    optimizer_name = trial.suggest_categorical(
        "optimizer", ["SGD", "Adam", "RMSprop", "Adagrad"]
    )
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)  # Changed the lower bound to 1e-5
    batch_size = trial.suggest_int("batch_size", 1, 512)
    n_epochs = trial.suggest_int("n_epochs", 10, 200)
    scheduler_name = trial.suggest_categorical(
        "scheduler",
        ["None", "StepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"],
    )

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
        output_activation = nn.Identity()  # Changed to use identity function for linear output

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
    net = Net(
        n_layers, n_units, hidden_activation, output_activation
    ).to(device)  # Added moving the network to the device

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
        # Creating the ReduceLROnPlateau scheduler with a threshold value of 0.01
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, threshold=0.01
        )
    else:
        scheduler = None

    # Returning all variables needed for saving and loading
    return net, loss_fn, optimizer, batch_size, n_epochs, scheduler,\
        loss_name, optimizer_name, scheduler_name,\
        n_units, n_layers,\
        hidden_activation,\
        output_activation



# ## The training and evaluation loop
# 
# We first define a couple of functions used in the training and evaluation.

# In[8]:


# Defining a function that computes primitive variables from conserved variables and predicted pressure
def compute_primitive_variables(x_batch, y_pred):
    """Computes primitive variables from conserved variables and predicted pressure using equations (A2) to (A5) from the paper.

    Args:
        x_batch (torch.Tensor): The input tensor of shape (batch_size, 3), containing conserved variables.
        y_pred (torch.Tensor): The predicted pressure tensor of shape (batch_size, 1).

    Returns:
        tuple: A tuple of (rho_pred, vx_pred, epsilon_pred), where rho_pred is rest-mass density,
            vx_pred is velocity in x-direction,
            epsilon_pred is specific internal energy,
            each being a torch tensor of shape (batch_size,).
    """
    # Extracting the conserved variables from x_batch
    D_batch = x_batch[:, 0]  # Conserved density
    Sx_batch = x_batch[:, 1]  # Conserved momentum in x-direction
    tau_batch = x_batch[:, 2]  # Conserved energy density

    # Computing the other primitive variables from y_pred and x_batch using equations (A2) to (A5) from the paper
    rho_pred = D_batch / torch.sqrt(1 + Sx_batch ** 2 / D_batch ** 2 / c ** 2)  # Rest-mass density
    vx_pred = Sx_batch / D_batch / c ** 2 / torch.sqrt(
        1 + Sx_batch ** 2 / D_batch ** 2 / c ** 2
    )  # Velocity in x-direction
    epsilon_pred = (
        tau_batch + D_batch
    ) / rho_pred - y_pred / rho_pred - 1  # Specific internal energy

    # Returning the primitive variables
    return rho_pred, vx_pred, epsilon_pred

# Defining a function that computes loss and metrics for a given batch
def compute_loss_and_metrics(y_pred, y_true, x_batch, loss_fn):
    """Computes loss and metrics for a given batch.

    Args:
        y_pred (torch.Tensor): The predicted pressure tensor of shape (batch_size, 1).
        y_true (torch.Tensor): The true pressure tensor of shape (batch_size,).
        x_batch (torch.Tensor): The input tensor of shape (batch_size, 3), containing conserved variables.
        loss_fn (torch.nn.Module or function): The loss function to use.

    Returns:
        tuple: A tuple of (loss, metric, rho_error, vx_error, epsilon_error), where loss is a scalar tensor,
            rho_error is relative error for rest-mass density,
            vx_error is relative error for velocity in x-direction,
            epsilon_error is relative error for specific internal energy,
            each being a scalar tensor.
    """
    # Reshaping the target tensor to match the input tensor
    y_true = y_true.view(-1, 1)

    # Computing the loss using the loss function
    loss = loss_fn(y_pred, y_true)

    # Computing the other primitive variables from y_pred and x_batch using the compute_primitive_variables function
    rho_pred, vx_pred, epsilon_pred = compute_primitive_variables(x_batch, y_pred)

    # Computing the true values of the other primitive variables from y_true and x_batch using the compute_primitive_variables function
    rho_true, vx_true, epsilon_true = compute_primitive_variables(x_batch, y_true)

    # Computing the relative errors for the other primitive variables
    rho_error = torch.mean(torch.abs((rho_pred - rho_true) / rho_true))  # Relative error for rest-mass density
    vx_error = torch.mean(torch.abs((vx_pred - vx_true) / vx_true))  # Relative error for velocity in x-direction
    epsilon_error = torch.mean(
        torch.abs((epsilon_pred - epsilon_true) / epsilon_true)
    )  # Relative error for specific internal energy

    # Returning the loss and metrics
    return loss, rho_error, vx_error, epsilon_error


# Defining a function that updates the learning rate scheduler with validation loss if applicable
def update_scheduler(scheduler, test_loss):
    """Updates the learning rate scheduler with validation loss if applicable.

    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): The learning rate scheduler to use.
        test_loss (float): The validation loss to use.

    Returns:
        None
    """
    # Checking if scheduler is not None
    if scheduler is not None:
        # Checking if scheduler is ReduceLROnPlateau
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # Updating the scheduler with test_loss
            scheduler.step(test_loss)
        else:
            # Updating the scheduler without test_loss
            scheduler.step()


# Now for the actual training and evaluation loop,

# In[9]:


# Defining a function to train and evaluate a network
def train_and_eval(net, loss_fn, optimizer, batch_size, n_epochs, scheduler, trial=None):
    """Trains and evaluates a network.

    Args:
        net (torch.nn.Module): The network to train and evaluate.
        loss_fn (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        batch_size (int): The batch size.
        n_epochs (int): The number of epochs.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        trial (optuna.trial.Trial): The trial object that contains the hyperparameters.
    Returns:
        float: The final metric value.
    """
    # Creating data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size
    )

    # Initializing lists to store the losses and metrics for each epoch
    train_losses = []
    test_losses = []
    train_metrics = []
    test_metrics = []

    # Creating a SummaryWriter object to log data for tensorboard
    writer = tbx.SummaryWriter()

    # Looping over the epochs
    for epoch in range(n_epochs):

        # Setting the network to training mode
        net.train()

        # Initializing variables to store the total loss and metrics for the train set
        train_loss = 0.0
        train_rho_error = 0.0
        train_vx_error = 0.0
        train_epsilon_error = 0.0

        # Looping over the batches in the train set
        for x_batch, y_batch in train_loader:

            # Moving the batch tensors to the device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Zeroing the gradients
            optimizer.zero_grad()

            # Performing a forward pass and computing the loss and metrics
            y_pred = net(x_batch)
            loss, rho_error, vx_error, epsilon_error = compute_loss_and_metrics(
                y_pred, y_batch, x_batch, loss_fn
            )


            # Performing a backward pass and updating the weights
            loss.backward()
            optimizer.step()

            # Updating the total loss and metrics for the train set
            train_loss += loss.item() * x_batch.size(0)
            train_rho_error += rho_error.item() * x_batch.size(0)
            train_vx_error += vx_error.item() * x_batch.size(0)
            train_epsilon_error += epsilon_error.item() * x_batch.size(0)

        # Computing the average loss and metrics for the train set
        train_loss /= len(train_loader.dataset)
        train_rho_error /= len(train_loader.dataset)
        train_vx_error /= len(train_loader.dataset)
        train_epsilon_error /= len(train_loader.dataset)

        # Appending the average loss and metrics for the train set to the lists
        train_losses.append(train_loss)
        train_metrics.append(
            {
                "rho_error": train_rho_error,
                "vx_error": train_vx_error,
                "epsilon_error": train_epsilon_error,
            }
        )

        # Logging the average loss and metrics for the train set to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Rho error/train", train_rho_error, epoch)
        writer.add_scalar("Vx error/train", train_vx_error, epoch)
        writer.add_scalar("Epsilon error/train", train_epsilon_error, epoch)

        # Setting the network to evaluation mode
        net.eval()

        # Initializing variables to store the total loss and metrics for the test set
        test_loss = 0.0
        test_rho_error = 0.0
        test_vx_error = 0.0
        test_epsilon_error = 0.0

        # Looping over the batches in the test set
        with torch.no_grad():
            for x_batch, y_batch in test_loader:

                # Moving the batch tensors to the device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Performing a forward pass and computing the loss and metrics
                y_pred = net(x_batch)
                loss, rho_error, vx_error, epsilon_error = compute_loss_and_metrics(
                    y_pred, y_batch, x_batch, loss_fn
                )


                # Updating the total loss and metrics for the test set
                test_loss += loss.item() * x_batch.size(0)
                test_rho_error += rho_error.item() * x_batch.size(0)
                test_vx_error += vx_error.item() * x_batch.size(0)
                test_epsilon_error += epsilon_error.item() * x_batch.size(0)

        # Computing the average loss and metrics for the test set
        test_loss /= len(test_loader.dataset)
        test_rho_error /= len(test_loader.dataset)
        test_vx_error /= len(test_loader.dataset)
        test_epsilon_error /= len(test_loader.dataset)

        # Appending the average loss and metrics for the test set to the lists
        test_losses.append(test_loss)
        test_metrics.append(
            {
                "rho_error": test_rho_error,
                "vx_error": test_vx_error,
                "epsilon_error": test_epsilon_error,
            }
        )

        # Logging the average loss and metrics for the test set to tensorboard
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Rho error/test", test_rho_error, epoch)
        writer.add_scalar("Vx error/test", test_vx_error, epoch)
        writer.add_scalar("Epsilon error/test", test_epsilon_error, epoch)

        # Printing the average loss and metrics for both sets for this epoch
        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
            f"Train Rho Error: {train_rho_error:.4f}, Test Rho Error: {test_rho_error:.4f}, "
            f"Train Vx Error: {train_vx_error:.4f}, Test Vx Error: {test_vx_error:.4f}, "
            f"Train Epsilon Error: {train_epsilon_error:.4f}, Test Epsilon Error: {test_epsilon_error:.4f}"
        )

        # Updating the learning rate scheduler with validation loss if applicable
        update_scheduler(scheduler, test_loss)

        # Reporting the intermediate metric value to Optuna if trial is not None
        if trial is not None:
            trial.report(metric.item(), epoch)

            # Checking if the trial should be pruned based on the intermediate value if trial is not None
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Closing the SummaryWriter object
    writer.close()

    # Returning the losses and metrics lists
    return train_losses, test_losses, train_metrics, test_metrics


# ## The objective function

# In[10]:


# Defining an objective function for Optuna to minimize
def objective(trial):
    """Defines an objective function for Optuna to minimize.

    Args:
        trial (optuna.trial.Trial): The trial object that contains the hyperparameters.

    Returns:
        float: The validation loss to minimize.
    """
    # Creating a trial network and optimizer using the create_model function
    net,\
    loss_fn,\
    optimizer,\
    batch_size,\
    n_epochs,\
    scheduler,\
    loss_name,\
    optimizer_name,\
    scheduler_name,\
    n_units,\
    n_layers,\
    hidden_activation,\
    output_activation = create_model(trial)


    # Training and evaluating the network using the train_and_eval function
    _, _, _, test_metrics = train_and_eval(
        net, loss_fn, optimizer, batch_size, n_epochs, scheduler, trial
    )

    # Returning the last validation epsilon error as the objective value to minimize
    return test_metrics[-1]["epsilon_error"]


# ## Finding the best parameters with Optuna

# In[11]:


# Creating a study object with Optuna with TPE sampler and Hyperband pruner
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner())

# Running Optuna with 100 trials without sampler and pruner arguments
study.optimize(objective, n_trials=100)

# Printing the best trial information
print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")


# ## Training with the best hyperparameters

# In[ ]:


# Creating the best network and optimizer using the best hyperparameters
net,\
loss_fn,\
optimizer,\
batch_size,\
n_epochs,\
scheduler,\
loss_name,\
optimizer_name,\
scheduler_name,\
n_units,\
n_layers,\
hidden_activation,\
output_activation = create_model(trial)


# Training and evaluating the best network using the train_and_eval function
train_losses, test_losses, train_metrics, test_metrics = train_and_eval(
    net, loss_fn, optimizer, batch_size, n_epochs, scheduler, trial
)


# ## Visualizing the results

# In[ ]:


# Plotting the losses and metrics for the best network
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.subplot(2, 2, 2)
plt.plot([m["rho_error"] for m in train_metrics], label="Train Rho Error")
plt.plot([m["rho_error"] for m in test_metrics], label="Test Rho Error")
plt.xlabel("Epoch")
plt.ylabel("Rho Error")
plt.legend()
plt.subplot(2, 2, 3)
plt.plot([m["vx_error"] for m in train_metrics], label="Train Vx Error")
plt.plot([m["vx_error"] for m in test_metrics], label="Test Vx Error")
plt.xlabel("Epoch")
plt.ylabel("Vx Error")
plt.legend()
plt.subplot(2, 2, 4)
plt.plot([m["epsilon_error"] for m in train_metrics], label="Train Epsilon Error")
plt.plot([m["epsilon_error"] for m in test_metrics], label="Test Epsilon Error")
plt.xlabel("Epoch")
plt.ylabel("Epsilon Error")
plt.legend()
plt.tight_layout()
plt.show()


# ## Saving

# In[ ]:


import pickle
import pandas as pd

# Saving the best network state dictionary using torch.save
torch.save(net.state_dict(), "best_net.pth")

# Saving the loss function name using pickle
with open("loss_fn.pkl", "wb") as f:
    pickle.dump(loss_name, f)

# Saving the optimizer name and parameters using pickle
with open("optimizer.pkl", "wb") as f:
    pickle.dump((optimizer_name, optimizer.state_dict()), f)

# Saving the best number of epochs using pickle
with open("n_epochs.pkl", "wb") as f:
    pickle.dump(n_epochs, f)

# Saving the scheduler name and parameters using pickle
with open("scheduler.pkl", "wb") as f:
    pickle.dump((scheduler_name, scheduler.state_dict()), f)

# Saving the number of units for each hidden layer using pickle
with open("n_units.pkl", "wb") as f:
    pickle.dump(n_units, f)

# Saving the output of create_model using pickle
with open("create_model.pkl", "wb") as f:
    pickle.dump((net, loss_fn, optimizer, batch_size, n_epochs, scheduler), f)

# Saving the output of the training using pandas
train_df = pd.DataFrame(
    {
        "train_loss": train_losses,
        "test_loss": test_losses,
        "train_rho_error": [m["rho_error"] for m in train_metrics],
        "test_rho_error": [m["rho_error"] for m in test_metrics],
        "train_vx_error": [m["vx_error"] for m in train_metrics],
        "test_vx_error": [m["vx_error"] for m in test_metrics],
        "train_epsilon_error": [m["epsilon_error"] for m in train_metrics],
        "test_epsilon_error": [m["epsilon_error"] for m in test_metrics],
    }
)
train_df.to_csv("train_output.csv", index=False)


# ## Loading

# In[ ]:


## Loading the best network state dictionary using torch.load
state_dict = torch.load("best_net.pth")

# Loading the state dictionary into a new network instance using net.load_state_dict
new_net = Net(n_layers, n_units, hidden_activation, output_activation).to(device)
new_net.load_state_dict(state_dict)


# In[ ]:


# Loading the loss function name using pickle
with open("loss_fn.pkl", "rb") as f:
    loss_name = pickle.load(f)

# Loading the optimizer name and parameters using pickle
with open("optimizer.pkl", "rb") as f:
    optimizer_name, optimizer_state_dict = pickle.load(f)

# Loading the best number of epochs using pickle
with open("n_epochs.pkl", "rb") as f:
    n_epochs = pickle.load(f)

# Loading the scheduler name and parameters using pickle
with open("scheduler.pkl", "rb") as f:
    scheduler_name, scheduler_state_dict = pickle.load(f)

# Loading the number of units for each hidden layer using pickle
with open("n_units.pkl", "rb") as f:
    n_units = pickle.load(f)

# Loading the output of create_model using pickle
with open("create_model.pkl", "rb") as f:
    net, loss_fn, optimizer, batch_size, n_epochs, scheduler = pickle.load(f)

# Loading the output of the training using pandas
train_df = pd.read_csv("train_output.csv")
train_losses = train_df["train_loss"].tolist()
test_losses = train_df["test_loss"].tolist()
train_metrics = [
    {
        "rho_error": train_df["train_rho_error"][i],
        "vx_error": train_df["train_vx_error"][i],
        "epsilon_error": train_df["train_epsilon_error"][i],
    }
    for i in range(len(train_df))
]
test_metrics = [
    {
        "rho_error": train_df["test_rho_error"][i],
        "vx_error": train_df["test_vx_error"][i],
        "epsilon_error": train_df["test_epsilon_error"][i],
    }
    for i in range(len(train_df))
]


# ## Testing

# In[ ]:


get_ipython().run_line_magic('load_ext', 'ipython_unittest')


# In[ ]:


get_ipython().run_cell_magic('unittest_main', '', '\n# Importing the unittest module\nimport unittest\n\n# Defining a class for testing the network and functions\nclass TestNet(unittest.TestCase):\n\n    # Defining a setup method to create a network and data\n    def setUp(self):\n        # Creating a network with fixed hyperparameters\n        self.net = Net(\n            n_layers=3,\n            n_units=[64, 32, 16],\n            hidden_activation=nn.ReLU(),\n            output_activation=nn.ReLU()\n        ).to(device)\n\n        # Creating a loss function\n        self.loss_fn = nn.MSELoss()\n\n        # Creating a batch of input and output data\n        self.x_batch = generate_input_data(10)\n        self.y_batch = generate_labels(10)\n\n    # Defining a test method to check if the network output has the correct shape\n    def test_net_output_shape(self):\n        # Performing a forward pass on the input data\n        y_pred = self.net(self.x_batch)\n\n        # Checking if the output shape matches the expected shape\n        self.assertEqual(y_pred.shape, (10, 1))\n\n    # Defining a test method to check if the eos_analytic function returns the correct pressure\n    def test_eos_analytic(self):\n        # Creating some sample primitive variables\n        rho = torch.tensor([1.0, 2.0, 3.0])\n        epsilon = torch.tensor([0.5, 1.0, 1.5])\n\n        # Computing the pressure using the eos_analytic function\n        p = eos_analytic(rho, epsilon)\n\n        # Checking if the pressure matches the expected values\n        self.assertTrue(torch.allclose(p, torch.tensor([0.6667, 2.6667, 6.0000]), atol=1e-4))\n\n    # Defining a test method to check if the compute_conserved_variables function returns the correct values\n    def test_compute_conserved_variables(self):\n        # Creating some sample primitive variables\n        rho = torch.tensor([1.0, 2.0, 3.0])\n        vx = torch.tensor([0.1 * c, 0.2 * c, 0.3 * c])\n        epsilon = torch.tensor([0.5, 1.0, 1.5])\n\n        # Computing the conserved variables using the compute_conserved_variables function\n        D, Sx, tau = compute_conserved_variables(rho, vx, epsilon)\n\n        # Checking if the conserved variables match the expected values\n        self.assertTrue(torch.allclose(D, torch.tensor([1.0050, 2.0202, 3.0469]), atol=1e-4))\n        self.assertTrue(torch.allclose(Sx, torch.tensor([0.1005, 0.4041, 0.9188]), atol=1e-4))\n        self.assertTrue(torch.allclose(tau, torch.tensor([0.6716, 3.3747, 8.1099]), atol=1e-4))\n\n    # Defining a test method to check if the compute_primitive_variables function returns the correct values\n    def test_compute_primitive_variables(self):\n        # Creating some sample conserved variables and predicted pressure\n        x_batch = torch.tensor([[1.0050, 0.1005, 0.6716], [2.0202, 0.4041, 3.3747], [3.0469, 0.9188, 8.1099]])\n        y_pred = torch.tensor([[0.6667], [2.6667], [6.0000]])\n\n        # Computing the primitive variables using the compute_primitive_variables function\n        rho_pred, vx_pred, epsilon_pred = compute_primitive_variables(x_batch, y_pred)\n\n        # Checking if the primitive variables match the expected values\n        self.assertTrue(torch.allclose(rho_pred, torch.tensor([1.0000, 2.0000, 3.0000]), atol=1e-4))\n        self.assertTrue(torch.allclose(vx_pred / c , torch.tensor([0.1000 , 0.2000 , 0.3000 ]), atol=1e-4))\n        self.assertTrue(torch.allclose(epsilon_pred , torch.tensor([0.5000 , 1.0000 , 1.5000 ]), atol=1e-4))\n\n    # Following function is to test if the loss function and metrics are computed correctly and that the\n    # torch tensors don’t have different shapes for y_pred and y_true in my compute_loss_and_metrics\n    # function.\n    # Defining a test method to check if the compute_loss_and_metrics function returns the correct values\n    def test_compute_loss_and_metrics(self):\n        # Creating some sample predicted and true pressure values\n        y_pred = torch.tensor([[0.6667], [2.6667], [6.0000]])\n        y_true = torch.tensor([[0.7000], [2.5000], [5.8000]])\n\n        # Computing the loss and metric using the compute_loss_and_metrics function\n        loss, metric = compute_loss_and_metrics(y_pred, y_true)\n\n        # Checking if the loss and metric match the expected values\n        self.assertTrue(torch.allclose(loss, torch.tensor(0.0167), atol=1e-4))\n        self.assertTrue(torch.allclose(metric, torch.tensor(0.0329), atol=1e-4))\n\n    # Defining a test method to check if the compute_loss_and_metrics function raises an error when y_pred and y_true have different shapes\n    def test_compute_loss_and_metrics_error(self):\n        # Creating some sample predicted and true pressure values with different shapes\n        y_pred = torch.tensor([[0.6667], [2.6667], [6.0000]])\n        y_true = torch.tensor([0.7000, 2.5000, 5.8000])\n\n        # Asserting that an error is raised when calling the compute_loss_and_metrics function with different shapes\n        self.assertRaises(ValueError, compute_loss_and_metrics, y_pred, y_true)\n\n')

