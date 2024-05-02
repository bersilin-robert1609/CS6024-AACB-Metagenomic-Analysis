import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

DATA_FOLDER = "../data"
ABUNDANCE_FILE = "abundance_with_unique_diff.csv"
ABUNDANCE_START = 212

OUT_FILE = '../logs/results-individual.json'

data = pd.read_csv(os.path.join(DATA_FOLDER, ABUNDANCE_FILE), low_memory=False)

dataset_names = data['dataset_name'].unique()

data_split = {}

# Remove spaces in the content of the 'disease' column
data['disease'] = data['disease'].str.strip()

for dataset_name in dataset_names:
    data_split[dataset_name] = data[data['dataset_name'] == dataset_name]

disease_info = {
    'Zeller_fecal_colorectal_cancer' : {'positive': ['cancer', 'small_adenoma', 'large_adenoma'], 'negative': ['n']},
    'WT2D': {'positive': ['t2d', 'impaired_glucose_tolerance'], 'negative': ['n']},
    'VerticalTransmissionPilot': {'positive': [], 'negative': ['nd']},
    't2dmeta_short': {'positive': ['t2d'], 'negative': ['n']},
    't2dmeta_long': {'positive': ['t2d'], 'negative': ['n', '-']},
    'Tito_subsistence_gut': {'positive': ['overweight', 'obese', 'underweight'], 'negative': ['n']},
    'Segre_Human_Skin': {'positive': [], 'negative': ['n']},
    'Quin_gut_liver_cirrhosis' : {'positive': ['cirrhosis'], 'negative': ['n']},
    'Psoriasis_2014': {'positive': ['y'], 'negative': ['n']},
    'Neilsen_genome_assembly': {'positive': ['ibd_ulcerative_colitis', 'ibd_crohn_disease'], 'negative': ['n', 'n_relative']},
    'metahit': {'positive': ['ibd_ulcerative_colitis', 'ibd_crohn_disease'], 'negative': ['n']},
    'Loman2013_EcoliOutbreak_DNA_MiSeq': {'positive': ['stec2-positive'], 'negative': []},
    'Loman2013_EcoliOutbreak_DNA_HiSeq': {'positive': ['stec2-positive'], 'negative': ['-']},
    'hmpii': {'positive': [], 'negative': ['n']},
    'hmp': {'positive': [], 'negative': ['n']},
    'Chatelier_gut_obesity': {'positive': ['obesity'], 'negative': ['n', 'leaness']},
    'Candela_Africa': {'positive': [], 'negative': ['n']},
}

data_subsets = {}

for disease in disease_info.keys():
    data_subset = data_split[disease]
    
    positive = pd.DataFrame()
    for pos in disease_info[disease]['positive']:
        subset = data_subset[data_subset['disease'] == pos]
        positive = pd.concat((positive, subset))
        
    negative = pd.DataFrame()
    for neg in disease_info[disease]['negative']:
        subset = data_subset[data_subset['disease'] == neg]
        negative = pd.concat((negative, subset))
        
    data_subsets[disease] = {'positive': positive, 'negative': negative}

IMAGES_FOLDER_DIFF = "../images/diff_features"

os.makedirs(IMAGES_FOLDER_DIFF, exist_ok=True)

def plot_barh(diff_mean_sorted, disease_name, n_count=20):
    """
    
    """
    top_features = diff_mean_sorted.index[:n_count]
    plt.figure(figsize=(10, 5))
    plt.barh(top_features, diff_mean_sorted[:n_count])
    plt.xlabel('Difference in mean abundance')
    plt.ylabel('Feature')
    plt.title(f'Top {n_count} features with highest mean difference for {disease_name} dataset')
    plt.savefig(os.path.join(IMAGES_FOLDER_DIFF, f'{disease_name}_top_features.png'), bbox_inches='tight')
    plt.close()
    

def preprocess_data(disease_name, data_subsets):
    data_subset = data_subsets[disease_name]
    
    positive_data = data_subset['positive']
    negative_data = data_subset['negative']
    
    # Only for visualizing ...
    
    positive_abundance = positive_data.iloc[:, ABUNDANCE_START:]
    negative_abundance = negative_data.iloc[:, ABUNDANCE_START:]
    
    positive_mean = positive_abundance.mean(axis=0)
    negative_mean = negative_abundance.mean(axis=0)
    
    diff_mean = abs(positive_mean - negative_mean)
    diff_mean_sorted = diff_mean.sort_values(ascending=False)
    
    if positive_data.shape[0] != 0 and negative_data.shape[0] != 0:
        plot_barh(diff_mean_sorted, disease_name)
    
    return positive_data, negative_data

# for disease_name in disease_info.keys():
#     preprocess_data(disease_name, data_subsets)

# Define the dataset class to load the data for the model

class DiseaseDataset(Dataset):
    def __init__(self, dataset_name, positive_data, negative_data, device):
        self.dataset_name = dataset_name
        self.positive_data = positive_data
        self.negative_data = negative_data
        
        self.data = pd.concat((positive_data, negative_data))
        
        self.labels = np.zeros(len(self.data))
        self.labels[:len(positive_data)] = 1 # positive samples are labeled as 1
        
        self.features = self.data.iloc[:, ABUNDANCE_START:]
        self.device = device
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X_train_tensor = torch.tensor(self.features.iloc[idx].values, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)
        
        return X_train_tensor, y_train_tensor

# Model

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

# Training for a epoch

def train(model: nn.Module, loader: DataLoader, optimizer: torch.optim, criterion: nn.Module):
    model.train()
    total_loss = 0.0
    
    for _, (X_train_tensor, y_train_tensor) in enumerate(loader):
        optimizer.zero_grad()
        
        if(len(y_train_tensor) < loader.batch_size):
            continue
        
        y_pred = model(X_train_tensor)
        
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    true = []
    pred = []
    
    with torch.no_grad():
        for _, (X_train_tensor, y_train_tensor) in enumerate(loader):
            y_pred = model(X_train_tensor)
            
            true.extend(y_train_tensor.cpu().numpy())
            pred.extend(y_pred.argmax(dim=1).cpu().numpy())
    
    return true, pred

hyper_parameters = {
    'disease': list(disease_info.keys()),
    'lr': [0.00001, 0.00005, 0.0001, 0.0005, 0.001],
    'batch_size': [4, 8, 16, 32],
    'layers': [[128, 64], [256, 128, 64], [512, 256, 128, 64]],
    'n_epochs': [200, 300, 400]
}

def train_model(dataset_name, data_subsets, LAYERS, lr, batch_size, n_epochs):
    
    DiseaseDataset_obj = DiseaseDataset(dataset_name, data_subsets[dataset_name]['positive'], data_subsets[dataset_name]['negative'], device)
    
    train_size = int(0.8 * len(DiseaseDataset_obj))
    test_size = len(DiseaseDataset_obj) - train_size
    
    train_dataset, test_dataset = random_split(DiseaseDataset_obj, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = MLP(DiseaseDataset_obj.features.shape[1], 2, LAYERS).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    train_losses = []
    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, optimizer, loss)
        train_losses.append(train_loss)
        
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {train_loss}')
    
    true_train, pred_train = evaluate(model, train_loader, loss)        
    true, pred = evaluate(model, test_loader, loss)
    
    return true_train, pred_train, true, pred, train_losses

# Get combinations of hyperparameters

import itertools
import json

hyper_parameters_values = list(itertools.product(*hyper_parameters.values()))

results = []

if(os.path.exists(OUT_FILE)):
    with open(OUT_FILE, 'r') as f:
        results = json.load(f)

for hyper_parameter_values in hyper_parameters_values:
    hyper_parameter = dict(zip(hyper_parameters.keys(), hyper_parameter_values))
    
    print(hyper_parameter)
    
    # check if already exists
    exists = False
    for result in results:
        if result['disease'] == hyper_parameter['disease'] and result['layers'] == hyper_parameter['layers'] and result['lr'] == hyper_parameter['lr'] and result['batch_size'] == hyper_parameter['batch_size'] and result['n_epochs'] == hyper_parameter['n_epochs']:
            exists = True
            print('Already exists')
            break
        
    if exists:
        continue
    
    disease = hyper_parameter['disease']
    if len(data_subsets[disease]['positive']) == 0 or len(data_subsets[disease]['negative']) == 0:
        continue
    
    true_train, pred_train, true, pred, train_loss = train_model(hyper_parameter['disease'], data_subsets, hyper_parameter['layers'], hyper_parameter['lr'], hyper_parameter['batch_size'], hyper_parameter['n_epochs'])
    
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary')
    
    results.append({
        'disease': hyper_parameter['disease'],
        'layers': hyper_parameter['layers'],
        'lr': hyper_parameter['lr'],
        'batch_size': hyper_parameter['batch_size'],
        'n_epochs': hyper_parameter['n_epochs'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_accuracy': accuracy_score(true_train, pred_train),
        'test_accuracy': accuracy_score(true, pred)
    })
    
    print("Train accuracy: ", results[-1]['train_accuracy'])
    print("Test accuracy: ", results[-1]['test_accuracy'])
    
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f)

print('Done')