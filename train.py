
import matplotlib.pyplot as plt
import math
import numpy as np
#  import cv2

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, confusion_matrix

seed = 2023
torch.manual_seed(seed)
np.random.seed(seed)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        #  self.conv1 = nn.Conv2d(3, 331, 335)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        #  self.fc1 = nn.Linear(41 * 83 * 64 * 2, 128) # two
        #  self.fc1 = nn.Linear(193600, 128) # three by three
        self.fc1 = nn.Linear(4096, 128) # three by three

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, images, labels=None):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx] if self.labels is not None else None
        return img, label


def load_dataset(filename): 

    dataset = None
    labels = None
    with np.load(dataset_npz) as data:
        #  print(data)
        #  print(data.files)
        #  dataset = data["dataset"]
        dataset = data["dataset"]
        labels = data["labels"]

    return dataset, labels


def binary_labels(labels): 

    for i in range(len(labels)): 
        if labels[i] != 0: 
            labels[i] = 1
    return labels

dataset_npz = "export/ZebraFish-03/6-by-6.npz"
images, labels = load_dataset(dataset_npz)

# processing
images = np.float32(images)
labels = binary_labels(labels)
classes = ("With Fish Head", "No Fish Head")

images = torch.from_numpy(images)
labels = torch.from_numpy(labels)

images = images.permute(0, 3, 1, 2)
dataset = CustomDataset(images, labels)
batch_size = 32

train_length = int(len(images) * 0.8)
test_length = len(images) - train_length
train_set, test_set = torch.utils.data.random_split(dataset, [train_length, test_length])
train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


cnn_model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

NUM_EPOCHS = 3
for epoch in range(NUM_EPOCHS): 
    #  running_loss = 0.0

    for i, data in enumerate(train_data_loader, 0): 
        inputs, labels = data
        optimizer.zero_grad()

        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #  running_loss += loss.item()
        #  if i % 100 == 0:    # print every 2000 mini-batches
        #      print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #      running_loss = 0.0


num_correct = 0
num_samples = 0

all_predictions = []
all_labels = []

cnn_model.eval()
with torch.no_grad(): 
    for x, y in test_data_loader: 
        scores = cnn_model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)

    print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}')

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Display confusion matrix using matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = ["With Fish Head", "No Fish Head"]
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


