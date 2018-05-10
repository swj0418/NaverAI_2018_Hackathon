import torch
from torch.utils.data import dataloader, dataset
from dataloader import Data_Loader
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(100)

Data = Data_Loader(csv_file='beers.csv', root_dir='H:/Datasets/craft-cans')
data2d = Data.Data_2D
print(data2d)

word_to_ix = {"A Word": 0, "Conversion": 1}
embeds = nn.Embedding(2, 5) # Embedding

lookup_tensor = torch.LongTensor([word_to_ix["A Word"]])
word_embed = embeds(autograd.Variable(lookup_tensor))
print(word_embed)

"""
This will be converted into a module later
"""

Container = []
percentage_idx = 1

for line_idx in range(len(data2d)):
    tmpContainer = []
    index = 0

    for elem_idx in range(len(data2d[line_idx])):
        elem = data2d[line_idx][elem_idx]
        word_to_ix = {elem: index}
        embeds = nn.Embedding(len(data2d) * len(data2d[0]), 10)
        lookup_tensor = torch.LongTensor([word_to_ix[elem]])
        index += 1
        embed = embeds(autograd.Variable(lookup_tensor))
        tmpContainer.append(embed)

    progress = (line_idx / (len(data2d) * len(data2d[0])) * 1000)
    if progress >= percentage_idx:
        percentage_idx += 1
        print("Progress : ", np.int32(progress), "% Completed")

    Container.append(tmpContainer)

print(Container)