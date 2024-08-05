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

def extract_boundary(nparray):
    array_dim = nparray.shape
    total_dimensions = len(array_dim)
    flat_boundary_values = np.array([[]])
    for d in range(total_dimensions):
        dimension_len = array_dim[d]
        boundary_array, unboundary_array = (
            np.take(nparray, [0, dimension_len - 1], axis=d),
            np.take(nparray, [x for x in range(dimension_len) if x not in [0, dimension_len - 1]], axis=d))
        flat_boundary_values = np.append(flat_boundary_values, boundary_array.flatten())
        nparray = unboundary_array
    return flat_boundary_values, unboundary_array

# def inject_data(nparray,add_interior = False, proportion = 0.1):
#     boundary_loader, interior_slices = torch.tensor([]), torch.tensor([])
#     for s in range(len(nparray)):
#         boundary, interior = extract_boundary(nparray[s][0])
#         boundary, interior = torch.tensor(boundary).float(), torch.tensor(interior).float()
#
#         interior_dims = nparray[s][0].shape
#         interior = torch.reshape(interior, [1,1]+[x for x in interior.shape])
#         interior_slices = torch.cat((interior_slices, interior), 0)
#
#         interior_elements_count = np.prod(interior.shape)
#         picked_elements = torch.tensor(np.random.choice(torch.flatten(interior[0][0]),size=int(proportion*interior_elements_count),replace=False))
#
#         picked_elements = torch.reshape(picked_elements, (1, 1, len(picked_elements), 1))
#
#         boundary = torch.reshape(boundary, (1, 1, len(boundary), 1))
#         if add_interior == True:
#             boundary = torch.cat((boundary, picked_elements), 2)
#         boundary_loader = torch.cat((boundary_loader, boundary), 0)
#
#     return boundary_loader

def inject_data(nparrays, proportion=1.0):
    boundary_loader_collection = torch.tensor([])
    picked_element_indexes, remainder_element_indexes = get_random_sample_indexes(nparrays[0][0], proportion=proportion)
    for nparray in nparrays:
        boundary_loader = torch.tensor([])
        for s in range(len(nparray)):
            boundary, interior = extract_boundary(nparray[s][0])
            boundary, interior = torch.tensor(boundary).float(), torch.tensor(interior).float()
            full_array = copy.deepcopy(nparray[s][0])

            full_array[1:len(full_array)-1,1:len(full_array)-1] = 0
            interior_dims = interior.shape


            picked_element_indexes.sort()
            picked_interior_elements = torch.flatten(interior)[picked_element_indexes]
            remainder_picked_elements = picked_interior_elements[remainder_element_indexes]

            sampled_interior_array = torch.cat((picked_interior_elements,remainder_picked_elements),dim=0)
            sampled_interior_array = torch.reshape(sampled_interior_array,interior_dims)
            full_array[1:len(full_array) - 1, 1:len(full_array) - 1] = sampled_interior_array

            full_array = torch.reshape(full_array, [1,1] + [x for x in full_array.shape])
            boundary_loader = torch.cat((boundary_loader, full_array), 0)
        boundary_loader = torch.reshape(boundary_loader, [1] + [x for x in boundary_loader.shape])
        boundary_loader_collection = torch.cat((boundary_loader_collection, boundary_loader), 0)





        # interior = torch.reshape(interior, [1, 1] + [x for x in interior.shape])
        # interior_slices = torch.cat((interior_slices, interior), 0)
        #
        # interior_elements_count = np.prod(interior.shape)
        # picked_elements = torch.tensor(
        #     np.random.choice(torch.flatten(interior[0][0]), size=int(proportion * interior_elements_count),
        #                      replace=False))
        #
        # picked_elements = torch.reshape(picked_elements, (1, 1, len(picked_elements), 1))
        #
        # boundary = torch.reshape(boundary, (1, 1, len(boundary), 1))
        # if add_interior == True:
        #     boundary = torch.cat((boundary, picked_elements), 2)
        # boundary_loader = torch.cat((boundary_loader, boundary), 0)

    return boundary_loader_collection

def get_random_sample_indexes(nparray, proportion = 1.0):
    picked_element_indexes = np.array([])
    remainder_element_indexes = np.array([])
    for s in range(len(nparray)):

        interior_array_dims = np.prod([x-2 if x > 1 else x for x in nparray.shape])
        element_count = np.prod(interior_array_dims)

        picked_indices = np.random.choice(element_count, max(1, int(proportion * element_count)), replace=False)
        picked_indices.sort()

        remainder = element_count - len(picked_indices)
        remainder_indices = np.random.choice(len(picked_indices), remainder, replace=True)

        picked_element_indexes = np.concatenate((picked_element_indexes, picked_indices), 0)
        remainder_element_indexes = np.concatenate((remainder_element_indexes, remainder_indices), 0)

        return picked_element_indexes, remainder_element_indexes

# %%
# Loading the Navier-Stokes dataset in 128x128 resolution

# train_loader, test_loaders, data_processor = load_darcy_flow_small(
#         n_train=1000, batch_size=32,
#         test_resolutions=[16, 32], n_tests=[100, 50],
#         test_batch_sizes=[32, 32],
#         positional_encoding=True
# )
train_resolution = 16
test_resolution = 32
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=100, batch_size=32,
        train_resolution=train_resolution,
        test_resolutions=[test_resolution], n_tests=[100],
        test_batch_sizes=[32],
        positional_encoding=True,
        download=True
)

# randomly pick indexes of a sample


train_loader.dataset.y, train_loader.dataset.x = train_loader.dataset.x, train_loader.dataset.y
# test_loaders[16].dataset.y, test_loaders[16].dataset.x = test_loaders[16].dataset.x, test_loaders[16].dataset.y
test_loaders[test_resolution].dataset.y, test_loaders[test_resolution].dataset.x = test_loaders[test_resolution].dataset.x, test_loaders[test_resolution].dataset.y

train_loader_original = copy.deepcopy(train_loader)
test_loaders_original = copy.deepcopy(test_loaders)

train_loader.dataset.x, train_loader.dataset.y = inject_data((train_loader.dataset.x,train_loader.dataset.y), proportion=1.0)
# test_loaders[16].dataset.x, test_loaders[16].dataset.y = inject_data((test_loaders[16].dataset.x,test_loaders[16].dataset.y), proportion=0.9)
# test_loaders[test_resolution].dataset.x, test_loaders[test_resolution].dataset.y = inject_data((test_loaders[test_resolution].dataset.x,test_loaders[test_resolution].dataset.y), proportion=1.0)
#
# test_loaders[32].dataset.x = inject_data(test_loaders[32].dataset.x, proportion=1.0)
# test_loaders[32].dataset.y = inject_data(test_loaders[32].dataset.y, proportion=1.0)


# train_loader.dataset.y = train_loader.dataset.x
# train_loader.dataset.x = train_loader.dataset.y
#
# test_loaders[32].dataset.x = test_loaders[32].dataset.y
# test_loaders[32].dataset.y = test_loaders[32].dataset.x

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
                  n_epochs=100,
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

fig = plt.figure(figsize=(15, 9))
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

    ax = fig.add_subplot(3, 5, index*5 + 1)
    ax.imshow(x[0], cmap='cubehelix')
    plot = ax.pcolor(x[0])
    fig.colorbar(plot)
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    offset = 0.3

    ax = fig.add_subplot(3, 5, index*5+ 2)
    ax.imshow(y.squeeze(),cmap='cubehelix')
    plot = ax.pcolor(y.squeeze(), vmin = min(y.squeeze().flatten())-offset, vmax = max(y.squeeze().flatten())+offset)
    fig.colorbar(plot)
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 3)
    ax.imshow(out.squeeze().detach().numpy(), cmap='cubehelix')
    plot = ax.pcolor(out.squeeze().detach().numpy(), vmin = min(y.squeeze().flatten())-offset, vmax = max(y.squeeze().flatten())+offset)
    fig.colorbar(plot)
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 5, index*5 + 4)
    ax.imshow(y.squeeze()-out.squeeze().detach().numpy(), cmap='cubehelix')
    plot = ax.pcolor(y.squeeze()-out.squeeze().detach().numpy(), vmin = min(y.squeeze().flatten())-offset, vmax = max(y.squeeze().flatten())+offset)
    fig.colorbar(plot)
    if index == 0:
        ax.set_title('pred - truth')
    plt.xticks([], [])
    plt.yticks([], [])

ax = fig.add_subplot(1, 5,5)
loss_array = np.loadtxt(f'./output/{current_date}/loss_file.txt')
ax.plot([x for x in range(len(loss_array))], loss_array)
ax.set_title('Loss plot')
ax.set_xlabel('epoch')
ax.set_xlabel('loss')
ax.set_yscale('log')
# ax.axis([1,len(loss_array),10**(-4),max(loss_array)*10])

# fig.subplots_adjust(wspace=1.0)


fig.suptitle('Inputs, ground-truth output, prediction, and difference.', y=0.98)
plt.tight_layout()
fig.show()
fig.savefig(f'./output/{current_date}/fig.png')

