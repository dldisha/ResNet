import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from ResNet import ResNet

hyperparams = {
    'N': 4,
    'C_1': 64,
    'P': 1,
    'B': [2, 2, 2, 2],
    'F': [3, 3, 3, 3],
    'K': [1, 1, 1, 1],
}

params = {
    'batch_size': 64,
    'lr': 0.01,
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

print(len(trainset))
print(len(testset))

# net = ResNet(**hyperparams)
net = torchvision.models.resnet18()
if params['gpu']:
    net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=params['lr'])

metrics = dict()
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
            metrics[iter_] = dict()
            metrics[iter_]['iter'] = iter_
            metrics[iter_]['epoch'] = epoch
            metrics[iter_]['train'] = loss.item()

            test_loss = 0
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
                    test_loss += criterion(outputs, labels).item() * labels.size(0)
            metrics[iter_]['test_loss'] = test_loss / total
            metrics[iter_]['test_acc'] = correct / total
            print(iter_, metrics[iter_])
#
# import matplotlib.pyplot as plt
# import pandas as pd
#
# metrics = pd.DataFrame(metrics).T
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(metrics['iter'], metrics['train'], label='train')
# plt.plot(metrics['iter'], metrics['test_loss'], label='test')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.plot(metrics['iter'], metrics['test_acc'], label='test')
# plt.xlabel('iteration')
# plt.ylabel('accuracy')
# plt.legend()
#
# plt.show()
#
# correct = 0
# total = 0
# with torch.no_grad():
#     for test_data in testloader:
#         inputs, labels = test_data
#         inputs = inputs.view(inputs.size(0), -1)
#         if params['gpu']:
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#         outputs = net(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print('Final Accuracy:', correct / total)
