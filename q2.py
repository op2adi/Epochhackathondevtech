import numpy as np
import torch
from sklearn.datasets import fetch_california_housing


def dataset():
    k = fetch_california_housing(as_frame=True)
    data = k.frame[['Longitude', 'Latitude']]
    return data.values


def distance_to_line(p, a, b):
    identity = np.identity(len(a))
    return np.linalg.norm((identity - np.dot(a, a.T)).dot(p - b)) ** 2

import torch


def cost_function(point_set, a, b):
    point_set = torch.tensor(point_set, dtype=torch.float)
    a = torch.tensor(a, dtype=torch.float, requires_grad=True)
    b = torch.tensor(b, dtype=torch.float, requires_grad=True)

    distances = []
    for i in point_set:
        distances.append(distance_to_line(i, a.detach().numpy(), b.detach().numpy()))
    max_distance = torch.max(torch.tensor(distances))
    return max_distance

# Assuming you have defined distance_to_line() function elsewhere



def gradient_descent(point_set, alpha=0.1, max_iterations=1000):
    a = torch.rand(2, requires_grad=True)
    a.data = a.data / torch.norm(a.data)
    b = torch.tensor([np.mean(point_set[:, 0]), np.mean(point_set[:, 1])], requires_grad=True)

    point_set = torch.tensor(point_set, dtype=torch.float)

    optimizer = torch.optim.Adam([a, b], lr=alpha)
    cost = []
    for i in range(max_iterations):
        # Zero the gradients
        optimizer.zero_grad()

        # Compute the cost function
        cost.append(cost_function(point_set, a, b))
        
    print(min(cost))

    return a.detach().numpy(), b.detach().numpy()

point_set = dataset()
fair_line = gradient_descent(point_set)
print(fair_line)
