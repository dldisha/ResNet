import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random
from ResNet import ResNet
import json
from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_parameter(hp):
    net = ResNet(**hp)
    params_size = count_parameters(net) / 1000000.
    return params_size < 5.


if __name__ == '__main__':
    feasible = []
    hyperparams = {
        'N': 3,
        'C_1': 64,
        'P': 1,
        'B': [2, 2, 2, 2],
        'F': [3, 3, 3, 3],
        'K': [1, 1, 1, 1],
    }
    count = 0
    for N in [2, 3, 4, 5]:
        for C_1 in [16, 32, 64, 128]:
            for P in [1, 2]:
                for B in [1, 2, 3]:
                    hyperparams['N'] = N
                    hyperparams['C_1'] = C_1
                    hyperparams['P'] = P
                    hyperparams['B'] = [B for _ in range(N)]
                    hyperparams['F'] = [3 for _ in range(N)]
                    hyperparams['K'] = [1 for _ in range(N)]
                    if get_parameter(hyperparams):
                        count += 1
                        feasible.append(hyperparams)
                        print(count)
                        print(N, C_1, P)
    random.shuffle(feasible)
    feasible = {i: item for i, item in enumerate(feasible)}
    with open('parameters.json', 'w') as f:
        json.dump(feasible, f)
