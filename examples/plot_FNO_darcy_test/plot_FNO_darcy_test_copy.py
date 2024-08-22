"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
"""
import copy
import os
import datetime
# %%
# 

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import random
import scipy as sp
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

print(torch.cuda.is_available())
# torch.cuda.current_device()
# torch.cuda.device(0)
# torch.cuda.get_device_name(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_date = datetime.datetime.now().strftime('%Y_%b_%d_%H_%M_%S')
if not os.path.exists('./output'): os.mkdir('./output')
if not os.path.exists(f'./output/{current_date}'): os.mkdir(f'./output/{current_date}')

sys.stdout = open(f'./output/{current_date}/log_file.txt', 'w')

# %%
# Loading the Navier-Stokes dataset in 128x128 resolution

# train_loader, test_loaders, data_processor = load_darcy_flow_small(
#         n_train=1000, batch_size=32,
#         test_resolutions=[16, 32], n_tests=[100, 50],
#         test_batch_sizes=[32, 32],
#         positional_encoding=True
# )
train_resolution = 16

test_resolution = 16
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        train_resolution=train_resolution,
        test_resolutions=[test_resolution], n_tests=[100,50],
        test_batch_sizes=[32,32],
        positional_encoding=True,
        download=True
)

# randomly pick indexes of a sample


train_loader.dataset.y, train_loader.dataset.x = train_loader.dataset.x, train_loader.dataset.y
test_loaders[test_resolution].dataset.y, test_loaders[test_resolution].dataset.x = test_loaders[test_resolution].dataset.x, test_loaders[test_resolution].dataset.y




train_loader_original = copy.deepcopy(train_loader)
test_loaders_original = copy.deepcopy(test_loaders)



data_processor = data_processor.to(device)


# %%
# We create a tensorized FNO model

# model = TFNO(n_modes=(16, 16),
#              hidden_channels=32,
#              projection_channels=64,
#              factorization='tucker',
#              rank=0.42,
#              in_channels=3,
#              out_channels=1
#              )
model = TFNO(n_modes=(16,16),
             hidden_channels=32,
             projection_channels=64,
             factorization='tucker',
             rank=0.42,
             # in_channels=3,
             # out_channels=1
             )
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
# optimizer = torch.optim.Adam(model.parameters(),
#                                 lr=8e-3,
#                                 weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
optimizer = torch.optim.LBFGS(model.parameters(),
                                lr=8e-3,
                                max_iter=20,
                              history_size=100)
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
                  n_epochs=300,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  log_test_interval=10,
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

os.rename(f'./output/loss_file.txt', f'./output/{current_date}/loss_file.txt')

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

test_samples = test_loaders_original[test_resolution].dataset

fig = plt.figure(figsize=(8, 8))


for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    #send to cpu for plotting
    x, y, out = x.cpu(), y.cpu(), out.cpu()

    # print(out)

    ax = fig.add_subplot(4, 4, index*4 + 1)
    ax.imshow(x[0], cmap='cubehelix')
    plot = ax.pcolor(x[0])
    fig.colorbar(plot)
    if index == 0:
        ax.set_title('Input u')
    plt.xticks([], [])
    plt.yticks([], [])

    offset = 0.3

    ax = fig.add_subplot(4, 4, index*4+ 2)
    ax.imshow(y.squeeze(),cmap='cubehelix')
    plot = ax.pcolor(y.squeeze(), vmin = min(y.squeeze().flatten())-offset, vmax = max(y.squeeze().flatten())+offset)
    fig.colorbar(plot)
    if index == 0:
        ax.set_title('Ground-truth a')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(4, 4, index*4 + 3)
    ax.imshow(out.squeeze().detach().numpy(), cmap='cubehelix')
    plot = ax.pcolor(out.squeeze().detach().numpy(), vmin = min(y.squeeze().flatten())-offset, vmax = max(y.squeeze().flatten())+offset)
    fig.colorbar(plot)
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])



    ax = fig.add_subplot(4, 4, index*4 + 4)
    diff = y[0].squeeze()-out.squeeze().detach().numpy()
    ax.imshow(np.abs(diff), cmap='cubehelix')
    plot = ax.pcolor(diff, vmin = torch.min(diff)-offset, vmax =  torch.max(diff)+offset)
    fig.colorbar(plot)
    if index == 0:
        ax.set_title('abs(pred - truth)')
    plt.xticks([], [])
    plt.yticks([], [])

ax = fig.add_subplot(4, 1,4)
loss_array = np.loadtxt(f'./output/{current_date}/loss_file.txt')
ax.plot([x for x in range(len(loss_array))], loss_array)
ax.set_title('Loss plot')
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_yscale('log')
ax.axis([1,len(loss_array),10**(-4),max(loss_array)*10])

# fig.subplots_adjust(wspace=1.0)


fig.suptitle('Inputs, ground-truth output, prediction, and difference.', y=0.98)
plt.tight_layout()
fig.show()
fig.savefig(f'./output/{current_date}/fig.png')

sys.stdout.close()

