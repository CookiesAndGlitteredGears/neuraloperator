"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
"""

# %%
# 

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss


device = 'cpu'

def extract_boundary(nparray):
    array_dim = nparray.shape
    total_dimensions = len(array_dim)
    flat_boundary_values = np.array([[]])
    for d in range(total_dimensions):
        dimension_len = array_dim[d]
        boundary_array, unboundary_array = (
            np.take(nparray, [0, dimension_len - 1], axis=d),
            np.take(nparray, [x for x in range(dimension_len) if x not in [0,dimension_len - 1]],axis=d))
        flat_boundary_values = np.append(flat_boundary_values, boundary_array.flatten())
        array = unboundary_array
    return flat_boundary_values

def inject_data(nparray):
    sample_loader = torch.tensor([])
    for s in range(len(nparray)):
        slice = torch.tensor(extract_boundary(nparray[s][0])).float()
        slice = torch.reshape(slice, (1, 1, len(slice), 1))
        sample_loader = torch.cat((sample_loader, slice), 0)
    return sample_loader
# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=10000, batch_size=128,
        test_resolutions=[32, 32], n_tests=[1000, 500],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)
train_loader.dataset.y, train_loader.dataset.x = train_loader.dataset.x, train_loader.dataset.y
# test_loaders[16].dataset.x, test_loaders[16].dataset.y = test_loaders[16].dataset.y, test_loaders[16].dataset.x

# test_loaders[32].dataset.x, test_loaders[32].dataset.y = test_loaders[32].dataset.y, test_loaders[32].dataset.x
test_loaders[32].dataset.x = inject_data(test_loaders[32].dataset.y)
test_loaders[32].dataset.y = inject_data(test_loaders[32].dataset.x)

# sample_loader = torch.tensor([])
# for s in range(len(test_loaders[32].dataset.x)):
#     slice = test_loaders[32].dataset.x[s][0]
#     slice = torch.tensor(extract_boundary(slice))
#     slice = torch.reshape(slice, (1, 1, len(slice), 1))
#     sample_loader = torch.cat((sample_loader, slice), 0)


data_processor = data_processor.to(device)


# %%
# We create a tensorized FNO model

model = TFNO(n_modes=(16, 16),
             hidden_channels=32,
             projection_channels=64,
             factorization='tucker',
             rank=0.42,
             in_channels=3,
             out_channels=1
             )
model = model.to(device)

# model(torch.tensor([[1.0,2.0]]))

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
trainer = Trainer(model=model,
                  n_epochs=200,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)


# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='cubehelix')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze(),cmap='cubehelix')
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy(), cmap='cubehelix')
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
