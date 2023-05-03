#!/usr/bin/env python
# coding: utf-8

# # Approximating h
# 
# We use Optuna to do a type of Bayesian optimization of the hyperparameters of the model. We then train the model using these hyperparameters to approximate the function $h = x_3 \cdot x_1^{x_2}$

# Use this first cell to convert the notebook to a python script.

# In[ ]:


# Importing the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import optuna
import tensorboardX as tbx
import matplotlib.pyplot as plt


# In[ ]:


# Defining the function to approximate
def h(x1, x2, x3):
    """Returns the value of the function h(x1, x2, x3) = x3 * x1 ** x2.

    Args:
        x1 (float or array-like): The first input.
        x2 (float or array-like): The second input.
        x3 (float or array-like): The third input.

    Returns:
        float or array-like: The value of the function h at the given inputs.
    """
    return x3 * x1 ** x2

# Generating the data samples
nsamples = 10**5

x1min = x3min = -10
x1max = x3max = 10
x2min = 0
x2max = 5

np.random.seed(0)
x1 = np.random.uniform(x1min, x1max, size=nsamples)
x2 = np.random.uniform(x2min, x2max, size=nsamples)
x3 = np.random.uniform(x3min, x3max, size=nsamples)
y = h(x1, x2, x3)

# Converting the data to tensors
x1 = torch.from_numpy(x1).float()
x2 = torch.from_numpy(x2).float()
x3 = torch.from_numpy(x3).float()
y = torch.from_numpy(y).float()

# Stacking the inputs into a single tensor
x = torch.stack([x1, x2, x3], dim=1)

# Splitting the data into train and test sets
train_size = int(0.8 * len(x))
test_size = len(x) - train_size
x_train, x_test = torch.split(x, [train_size, test_size])
y_train, y_test = torch.split(y, [train_size, test_size])


# In[ ]:


# Defining a custom neural network class
class Net(nn.Module):
    """A custom neural network class that approximates the function h.

    Args:
        n_layers (int): The number of hidden layers in the network.
        n_units (int): The number of hidden units in each layer.
        hidden_activation (torch.nn.Module): The activation function for the hidden layers.
        output_activation (torch.nn.Module): The activation function for the output layer.
    """
    def __init__(self, n_layers, n_units, hidden_activation, output_activation):
        super(Net, self).__init__()
        # Creating a list of linear layers with n_units each
        self.layers = nn.ModuleList([nn.Linear(3, n_units)])
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_units, n_units))
        # Adding the final output layer with one unit
        self.layers.append(nn.Linear(n_units, 1))
        # Storing the activation functions
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def forward(self, x):
        """Performs a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 1).
        """
        # Passing the input through the hidden layers with the hidden activation function
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        # Passing the output through the final layer with the output activation function
        x = self.output_activation(self.layers[-1](x))
        return x


# In[ ]:


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
    n_layers = trial.suggest_int("n_layers", 1, 3)
    n_units = trial.suggest_int("n_units", 32, 128)
    hidden_activation_name = trial.suggest_categorical("hidden_activation", ["ReLU", "Sigmoid"])
    output_activation_name = trial.suggest_categorical("output_activation", ["ReLU", "Sigmoid"])
    loss_name = trial.suggest_categorical("loss", ["MSE", "MAE"])
    optimizer_name =  trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_int("batch_size", 64, 256)
    n_epochs = trial.suggest_int("n_epochs", 10, 100)
    scheduler_name = None # trial.suggest_categorical("scheduler", ["None", "StepLR", "ExponentialLR"])

    # Creating the activation functions from their names
    if hidden_activation_name == "ReLU":
        hidden_activation = nn.ReLU()
    elif hidden_activation_name == "Tanh":
        hidden_activation = nn.Tanh()
    else:
        hidden_activation = nn.Sigmoid()

    if output_activation_name == "ReLU":
        output_activation = nn.ReLU()
    elif output_activation_name == "Tanh":
        output_activation = nn.Tanh()
    else:
        output_activation = nn.Sigmoid()

    # Creating the loss function from its name
    if loss_name == "MSE":
        loss_fn = nn.MSELoss()
    elif loss_name == "MAE":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.SmoothL1Loss()

    # Creating the network with the sampled hyperparameters
    net = Net(n_layers, n_units, hidden_activation, output_activation)

    # Creating the optimizer from its name
    if optimizer_name == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=lr)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=lr)

    # Creating the learning rate scheduler from its name
    if scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_name == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        scheduler = None

    # Returning the network, the loss function, the optimizer, the batch size, the number of epochs and the scheduler
    return net, loss_fn, optimizer, batch_size, n_epochs, scheduler


# In[ ]:


# Defining a function to train and evaluate a network
def train_and_eval(net, loss_fn, optimizer, batch_size, n_epochs, scheduler):
    """Trains and evaluates a network on the train and test sets.

    Args:
        net (Net): The network to train and evaluate.
        loss_fn (torch.nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        batch_size (int): The batch size to use.
        n_epochs (int): The number of epochs to train for.
        scheduler (torch.optim.lr_scheduler._LRScheduler or None): The learning rate scheduler to use.

    Returns:
        tuple: A tuple of (train_losses, test_losses,
            train_accuracies, test_accuracies), where train_losses and test_losses are lists of average losses per epoch for the train and test sets,
            train_accuracies and test_accuracies are lists of average accuracies per epoch for the train and test sets.
    """
    # Creating data loaders for train and test sets
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size)

    # Initializing lists to store the losses and accuracies for each epoch
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Looping over the epochs
    for epoch in range(n_epochs):
        # Setting the network to training mode
        net.train()
        # Initializing variables to store the total loss and correct predictions for the train set
        train_loss = 0.0
        train_correct = 0
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
            # Updating the total loss and correct predictions
            train_loss += loss.item() * x_batch.size(0)
            train_correct += torch.sum((y_pred - y_batch).abs() < 1e-3)

        # Updating the learning rate scheduler if applicable
        if scheduler is not None:
            scheduler.step()

        # Setting the network to evaluation mode
        net.eval()
        # Initializing variables to store the total loss and correct predictions for the test set
        test_loss = 0.0
        test_correct = 0
        # Looping over the batches in the test set
        for x_batch, y_batch in test_loader:
            # Forward pass
            y_pred = net(x_batch)
            # Reshaping the target tensor to match the input tensor
            y_batch = y_batch.view(-1, 1)
            # Computing the loss
            loss = loss_fn(y_pred, y_batch)
            # Updating the total loss and correct predictions
            test_loss += loss.item() * x_batch.size(0)
            test_correct += torch.sum((y_pred - y_batch).abs() < 1e-3)

        # Computing the average losses and accuracies for the train and test sets
        train_loss /= len(x_train)
        test_loss /= len(x_test)
        train_accuracy = train_correct / len(x_train)
        test_accuracy = test_correct / len(x_test)

        # Appending the losses and accuracies to the lists
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Printing the epoch summary
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Returning the lists of losses and accuracies
    return train_losses, test_losses, train_accuracies, test_accuracies


# In[ ]:


# Defining a function to compute the objective value for a trial
def objective(trial):
    """Computes the objective value (final test loss) for a trial.

    Args:
        trial (optuna.trial.Trial): The trial object that contains the hyperparameters.

    Returns:
        float: The objective value (final test loss) for the trial.
    """
    # Creating a trial network and optimizer with the create_model function
    net, loss_fn, optimizer, batch_size, n_epochs, scheduler = create_model(trial)

    # Training and evaluating the network with the train_and_eval function
    train_losses, test_losses, train_accuracies, test_accuracies = train_and_eval(net, loss_fn, optimizer, batch_size, n_epochs, scheduler)

    # Returning the final test loss as the objective value
    return test_losses[-1]


# ## Finding the best hyperparameters with Optuna

# In[ ]:


# Creating a study object with Optuna
study = optuna.create_study(direction="minimize")

# Running the optimization for 100 trials
study.optimize(objective, n_trials=100)


# ## Training with the best parameters

# In[ ]:


# Printing the best parameters and the best value
print("Best parameters:", study.best_params)
print("Best value:", study.best_value)

# Creating a network with the best parameters
best_net, best_loss_fn, best_optimizer, best_batch_size, best_n_epochs, best_scheduler = create_model(study.best_trial)

# Training and evaluating the network with the best parameters
best_train_losses, best_test_losses, best_train_accuracies, best_test_accuracies = train_and_eval(best_net, best_loss_fn, best_optimizer, best_batch_size, best_n_epochs, best_scheduler)

# Saving the network with the best parameters
torch.save(best_net.state_dict(), "best_net.pth")


# ## Visualizing results

# In[ ]:


# Creating a tensorboard writer
writer = tbx.SummaryWriter()

# Logging the losses and accuracies for each epoch
for epoch in range(best_n_epochs):
    writer.add_scalars("Loss", {"Train": best_train_losses[epoch], "Test": best_test_losses[epoch]}, epoch + 1)
    writer.add_scalars("Accuracy", {"Train": best_train_accuracies[epoch], "Test": best_test_accuracies[epoch]}, epoch + 1)

# Closing the writer
writer.close()

# Plotting the predicted values and true values for a random sample of 1000 test points
indices = np.random.choice(len(x_test), size=1000)
x_sample = x_test[indices]
y_sample = y_test[indices]
y_pred = best_net(x_sample).detach().numpy()
plt.scatter(y_sample, y_pred)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted Values vs True Values")
plt.savefig("pred_vs_true.png")

# Plotting the bias (difference between predicted values and true values) for a random sample of 1000 test points
bias = y_pred - y_sample.numpy()
plt.hist(bias)
plt.xlabel("Bias")
plt.ylabel("Frequency")
plt.title("Bias Distribution")
plt.savefig("bias_dist.png")

