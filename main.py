import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from prettytable import PrettyTable
from multiprocessing import Pool

from ResNet import ResNet


def str2bool(v):
    return v.lower() in ('true', '1')


parser = argparse.ArgumentParser()

# -------------------------------------------------------------------------
# Data input settings
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=66)
# Load Models
start_end, unparsed = parser.parse_known_args()

params = {
    'batch_size': 64,
    'lr': 0.001,
    'workers': 0,
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'gpu': True,
    'epoch': 50,
}


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomCrop(size=32, padding=4),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params['batch_size'], shuffle=True,
                                          num_workers=params['workers'])

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=params['batch_size'], shuffle=False,
                                         num_workers=params['workers'])


def run(hp):
    if 'D_P' in hp:
        params['lr'] = 5e-4
        params['epoch'] = 60
    name = json.dumps(hp)
    if os.path.exists('outputs/' + name + '.json'):
        return 0
    net = ResNet(**hp)
    params_size = count_parameters(net) / 1000000.
    if params_size > 5.:
        print(params_size)
        raise AssertionError
    if params['gpu']:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=params['lr'])

    hp['score'] = 0.0
    hp['epoch'] = 0
    iter_ = 0
    best_model = None
    for epoch in range(params['epoch']):
        for i, data in enumerate(trainloader):
            iter_ += 1

            inputs, labels = data
            if params['gpu']:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_ % 100 == 99:
                correct = 0
                total = 0
                with torch.no_grad():
                    for test_data in testloader:
                        inputs, labels = test_data
                        if params['gpu']:
                            inputs = inputs.cuda()
                            labels = labels.cuda()
                        outputs = net(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                if hp['score'] < correct / total:
                    hp['score'] = correct / total
                    hp['epoch'] = epoch
                    best_model = net.state_dict()
    torch.save(best_model, 'outputs/' + name + '.pt')
    with open('outputs/' + name + '.json', 'w') as f:
        json.dump(hp, f)
    return hp


if __name__ == '__main__':
    # hyperparams = {
    #     'N': 3,
    #     'C_1': 64,
    #     'P': 1,
    #     'B': [2, 2, 2, 2],
    #     'F': [3, 3, 3, 3],
    #     'K': [1, 1, 1, 1],
    #     'D_P': 0.1,
    #     'D_S': 3
    # }
    with open("parameters.json", 'r') as f:
        candidates = json.load(f)
    candidates = [candidates[str(i)] for i in range(start_end.start, start_end.end)]
    with Pool(6) as p:
        p.map(run, candidates)
