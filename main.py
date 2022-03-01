import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from ResNet import ResNet
import json

params = {
    'batch_size': 128,
    'lr': 0.001,
    'workers': 0,
    'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'gpu': True,
    'epoch': 10,
}

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params['batch_size'], shuffle=True,
                                          num_workers=params['workers'])

testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=params['batch_size'], shuffle=False,
                                         num_workers=params['workers'])


def run(hp):
    name = json.dumps(hp)
    net = ResNet(**hp)
    if params['gpu']:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=params['lr'])

    hp['score'] = 0.0
    hp['epoch'] = 0
    iter_ = 0
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
                    torch.save(net.state_dict(), name + '.pt')
    with open(name + '.json', 'w') as f:
        json.dump(hp, f)
    return hp


if __name__ == '__main__':
    hyperparams = {
        'N': 4,
        'C_1': 64,
        'P': 1,
        'B': [2, 2, 2, 2],
        'F': [3, 3, 3, 3],
        'K': [1, 1, 1, 1],
    }
    run(hyperparams)
