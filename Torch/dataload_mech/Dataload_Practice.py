import torch
from torch.utils.data import dataloader, dataset, DataLoader

wordlist = ["첫번쨰 단어", "두번재 단어", "세번째 단어"]
label = [0, 1, 2]

data_set = wordlist, label

train_loader = DataLoader(data_set,
                          batch_size=2,
                          shuffle=False,
                          num_workers=1)

print(train_loader.dataset)