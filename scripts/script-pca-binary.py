import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_recall_fscore_support

DATA_FOLDER = "../data"
ABUNDANCE_FILE = "abundance_with_unique.csv"
ABUNDANCE_START = 212

PERCENT_INFO = 0.95
TEST_SPLIT = 0.2
SEED = 42

BATCH_SIZE = 32
LAYERS = [1024, 512, 256, 128, 64, 64, 32, 32]
LEARNING_RATE = 0.000001

abundance_data = pd.read_csv(os.path.join(DATA_FOLDER, ABUNDANCE_FILE), low_memory=False)

metadata = abundance_data.iloc[:, :ABUNDANCE_START]
abundance = abundance_data.iloc[:, ABUNDANCE_START:]

scalar = StandardScaler()
abundance_X_scaled = scalar.fit_transform(abundance)

pca = PCA()
abundance_X_pca = pca.fit_transform(abundance_X_scaled)

explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)
print(explained_variance_ratio_cumulative.shape)

plt.plot(explained_variance_ratio_cumulative)
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio vs Number of Components")

# Plot a line at 0.95 in the y-axis
plt.axhline(y=PERCENT_INFO, color='r', linestyle='--', label='95% Explained Variance')
plt.axvline(x=np.argmax(explained_variance_ratio_cumulative > PERCENT_INFO), color='g', linestyle='-', label='Required Components')
plt.legend(loc='best')
plt.show()

num_components = np.argmax(explained_variance_ratio_cumulative >= PERCENT_INFO) + 1

print("The number of components to explain {}% of the variance is {}".format(PERCENT_INFO * 100, num_components))
print("The percent of num comps in total comps is {}".format(num_components / len(pca.explained_variance_ratio_) * 100))

abundance_top_components = abundance_X_pca[:, :num_components]
print(abundance_top_components.shape)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class AbundanceDataset(Dataset):
    def __init__(self, abundance_data, metadata):
        self.abundance_data = abundance_data
        self.metadata = metadata
        
        self.n_classes = len(self.metadata['disease'].unique())
        self.classes = self.metadata['disease'].unique()
        
        healthy_classes = ['n', 'leaness', 'nd', ' -', 'n_relative', '-']
        
        # Change the healthy classes to 'healthy'
        self.target = self.metadata['disease']
        
        self.target = self.target.apply(lambda x: 0 if x in healthy_classes else 1)
        self.target = self.target.values
        
        self.n_classes = len(np.unique(self.target))
        
        print("Healthy samples: ", len(self.target[self.target == 0]))
        print("Disease samples: ", len(self.target[self.target == 1]))
        print("Total samples: ", len(self.target))
        print("Total classes: ", self.n_classes)

    def __len__(self):
        return len(self.abundance_data)

    def __getitem__(self, idx):
        return torch.tensor(self.abundance_data[idx], dtype=torch.float32), torch.tensor(self.target[idx], dtype=torch.long)

from sklearn.model_selection import train_test_split

abundance_train, abundance_test, metadata_train, metadata_test = train_test_split(abundance_top_components, metadata, test_size=TEST_SPLIT, random_state=SEED)

dataset_train = AbundanceDataset(abundance_train, metadata_train)
dataset_test = AbundanceDataset(abundance_test, metadata_test)

class MLP(nn.Module):
    def __init__(self, idim, odim, layers, batch_norm=True):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        
        for i, layer in enumerate(layers):
            if i == 0:
                self.layers.add_module("fc{}".format(i), nn.Linear(idim, layer))
            else:
                self.layers.add_module("fc{}".format(i), nn.Linear(layers[i-1], layer))
            
            if batch_norm:
                self.layers.add_module("bn{}".format(i), nn.BatchNorm1d(layer))
            
            self.layers.add_module("relu{}".format(i), nn.ReLU())
            
        self.layers.add_module("fc{}".format(len(layers)), nn.Linear(layers[-1], odim))
        self.layers.add_module("softmax", nn.Softmax(dim=1))
        
    def forward(self, x):
        return self.layers(x)

# Training

def train(model, data_loader, criterion, optimizer):
    """
    Generic training function
    """
    model.train() # Set model to training mode
    running_loss = 0.0
    for _, data in enumerate(data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad() # Zero the gradients
        
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        
        running_loss += loss.item()
    return running_loss / len(data_loader)

def evaluate(model, data_loader, criterion):
    """
    Generic evaluation function    
    """
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return running_loss / len(data_loader), correct / total, true_labels, predicted_labels

def train_one_model(params):
    dataloader_train = DataLoader(dataset_train, batch_size=params['batch_size'], shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=params['batch_size'], shuffle=False)
    
    model = MLP(abundance_top_components.shape[1], dataset_train.n_classes, params['layers']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    for epoch in range(params['epochs']):
        train_loss = train(model, dataloader_train, criterion, optimizer)
        
        if(epoch % 100 == 0):
            print("Epoch: {} Train Loss: {}".format(epoch, train_loss))
        
    train_loss, train_accuracy, true_train, pred_train = evaluate(model, dataloader_train, criterion)
    test_loss, test_accuracy, true_test, pred_test = evaluate(model, dataloader_test, criterion)
    print("Test Loss: {} Test Accuracy: {}".format(test_loss, test_accuracy))
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_test, pred_test, average='binary')
    precision_train, recall_train, f1_train, _ = precision_recall_fscore_support(true_train, pred_train, average='binary')
    
    return train_loss, train_accuracy, test_loss, test_accuracy, precision, recall, f1, precision_train, recall_train, f1_train

hyperparameters = {
    'batch_size': [32],
    'layers': [
        [512, 256, 128, 64, 32],
        [1024, 512, 256, 128, 64, 64, 32, 32],
        [1024, 1024, 512, 512, 128, 64, 32]
    ],
    
    'learning_rate': [0.0001, 0.0005, 0.00001, 0.00005, 0.000001],
    'epochs': [100, 200, 300, 400, 500],
}

results = []

import os
import json

OUTPUT_FILE = "../logs/results-mlp-pca-binary.json"

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        results = json.load(f)
        
import itertools

for batch_size, layers, learning_rate, epochs in itertools.product(hyperparameters['batch_size'], hyperparameters['layers'], hyperparameters['learning_rate'], hyperparameters['epochs']):
    params = {
        'batch_size': batch_size,
        'layers': layers,
        'learning_rate': learning_rate,
        'epochs': epochs
    }
    
    print("Training model with params: ", params)
    
    # check if the model has already been trained
    found = False
    for result in results:
        if result['params'] == params:
            found = True
            break
        
    if found:
        print("Model already trained. Skipping...")
        continue
    
    results_one = train_one_model(params)
    results.append({
        'params': params,
        'train_loss': results_one[0],
        'train_accuracy': results_one[1],
        'test_loss': results_one[2],
        'test_accuracy': results_one[3],
        'precision': results_one[4],
        'recall': results_one[5],
        'f1': results_one[6],
        'precision_train': results_one[7],
        'recall_train': results_one[8],
        'f1_train': results_one[9]
    })
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f)




