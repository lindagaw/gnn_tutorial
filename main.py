import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import torch_geometric.transforms as T

from datasets import KarateDataset

dataset = KarateDataset()
data = dataset[0]

print(dataset)
print(data)
