> Explain L1 norm and Linf norm calculations in the context of my code

```python
def train_and_eval(net, loss_fn, optimizer, batch_size, n_epochs, scheduler, train_loader, val_loader, test_loader, trial=None):
  
    train_losses = []
    val_losses = []
    test_losses = []
    train_metrics = []
    val_metrics = []
    test_metrics = []
  
    for epoch in range(n_epochs):
      
        net.train()
      
        train_loss = 0.0
        train_l1_norm = 0.0
        train_linf_norm = 0.0
      
        for x_batch, y_batch in train_loader:
          
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
          
            optimizer.zero_grad()
          
            y_pred = net(x_batch)
            loss, l1_norm, linf_norm = compute_loss_and_metrics(y_pred, y_batch, loss_fn)
          
            loss.backward()
            optimizer.step()
          
            train_loss += loss.item() * x_batch.size(0)
            train_l1_norm += l1_norm.item() * x_batch.size(0)
            train_linf_norm += linf_norm.item() * x_batch.size(0)
      
        train_loss /= len(train_loader.dataset)
        train_l1_norm /= len(train_loader.dataset)
        train_linf_norm /= len(train_loader.dataset)
      
        train_losses.append(train_loss)
        train_metrics.append(
            {
                "l1_norm": train_l1_norm,
                "linf_norm": train_linf_norm,
            }
        )

        if val_loader is not None:
            net.eval()
            val_loss = 0.0
            val_l1_norm = 0.0
            val_linf_norm = 0.0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_pred = net(x_batch)
                    loss, l1_norm, linf_norm = compute_loss_and_metrics(y_pred, y_batch, loss_fn)

                    val_loss += loss.item() * x_batch.size(0)
                    val_l1_norm += l1_norm.item() * x_batch.size(0)
                    val_linf_norm += linf_norm.item() * x_batch.size(0)

            val_loss /= len(val_loader.dataset)
            val_l1_norm /= len(val_loader.dataset)
            val_linf_norm /= len(val_loader.dataset)

            val_losses.append(val_loss)
            val_metrics.append(
                {
                    "l1_norm": val_l1_norm,
                    "linf_norm": val_linf_norm,
                }
            )

            update_scheduler(scheduler, val_loss)

        if test_loader is not None:
            net.eval()
            test_loss = 0.0
            test_l1_norm = 0.0
            test_linf_norm = 0.0

            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_pred = net(x_batch)
                    loss, l1_norm, linf_norm = compute_loss_and_metrics(y_pred, y_batch, loss_fn)

                    test_loss += loss.item() * x_batch.size(0)
                    test_l1_norm += l1_norm.item() * x_batch.size(0)
                    test_linf_norm += linf_norm.item() * x_batch.size(0)

            test_loss /= len(test_loader.dataset)
            test_l1_norm /= len(test_loader.dataset)
            test_linf_norm /= len(test_loader.dataset)

            test_losses.append(test_loss)
            test_metrics.append(
                {
                    "l1_norm": test_l1_norm,
                    "linf_norm": test_linf_norm,
                }
            )
      
        if trial is not None:
            trial.report(val_l1_norm, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
  
    return train_losses, val_losses, test_losses, train_metrics, val_metrics, test_metrics

def compute_loss_and_metrics(y_pred, y_true, loss_fn):
    y_true = y_true.view(-1, 1)
    loss = loss_fn(y_pred, y_true)
    rel_error = torch.abs((y_pred - y_true) / y_true)
    l1_norm = torch.mean(rel_error) 
    linf_norm = torch.max(rel_error) 
    return loss, l1_norm, linf_norm
```

In your code, you are using L1 norm and Linf norm to measure the relative error between the predicted and true values of a regression problem. L1 norm and Linf norm are types of vector norms that measure the magnitude or distance of a vector.

L1 norm is the sum of the absolute values of the elements of a vector. It is also known as the Manhattan distance or the taxicab norm. In your code, you are using torch.mean() to compute the average L1 norm across all the samples in a batch. This gives you a measure of how close your predictions are to the true values on average.

Linf norm is the maximum absolute value of the elements of a vector. It is also known as the infinity norm or the maximum norm. In your code, you are using torch.max() to compute the maximum Linf norm across all the samples in a batch. This gives you a measure of how far your worst prediction is from the true value.

Both L1 norm and Linf norm can be used as metrics to evaluate the performance of a regression model. They can also be used as regularization terms to penalize large errors or deviations from the true values.

I hope this helps you understand what L1 norm and Linf norm are and how they are calculated in your code.😊

> How come the norms differ per batch?

In the given code, the L1 and Linf norms are calculated on a per-batch basis. The norms may differ per batch because the relative errors between the predicted and true values of the target variable can vary across batches.

Batch normalization is a commonly used technique in machine learning that normalizes the activations of the network layer by scaling and shifting them. The main purpose of batch normalization is to reduce the effects of internal covariate shift, which occurs when the distribution of the inputs to a layer changes during training. By normalizing the activations, batch normalization can help the model to stabilize training and improve its accuracy.

Normalization adjustments per batch can affect the L1 and Linf norms because the relative errors between the predicted and true values of the target variable change with each batch. Specifically, two batches from the same dataset but at different points in the training process could have different relative errors, leading to different L1 and Linf norms.

Therefore, the norms differ per batch because the relative errors used to compute them may vary depending on the normalization adjustments and other factors that affect the performance of the machine learning model during training.

> If batches are just different sample sets of the data, then why do the norms decrease over time?

The L1 and Linf norms calculated per-batch may decrease over time due to the updates made to the model weights during training. The model updates occur every time a batch is fed through the network, which leads to gradual changes in the network's output over time.

As the model sees more and more data, it can learn to make better predictions, resulting in a reduction in the L1 and Linf norms over time. This means that the model's relative errors between predicted and true values of the target variable decrease with each batch.

In addition to model updates, the decreasing trends in the L1 and Linf norms may also be influenced by other factors such as the optimizer, learning rate, and regularization applied during training. For example, a learning rate schedule may decrease the learning rate over time, leading to smaller updates to the model and a slower decrease in the norms.

Therefore, while batches are just different sample sets of the data, various factors such as model updates, optimizer settings, and regularization can have a cumulative effect on the model's performance over time. These factors can lead to a gradual decrease in the L1 and Linf norms as the model learns to make better predictions and reduces the relative errors between the predicted and true values of the target variable.