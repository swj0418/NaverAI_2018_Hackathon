import torch
import os
import sys
import numpy
import pandas as pd
import matplotlib as plt
from torch.utils.data import Dataset, DataLoader
import argparse

class Data_Loader(Dataset):
    Data_2D = []
    data_rowsize = 0
    data_colsize = 0

    def __init__(self, csv_file, root_dir, transform=None):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

        # Basic data loading
        self.load_file()

    def load_file(self):
        for line in open(os.path.join(self.root_dir, self.csv_file), encoding='utf-8', mode='r'):
            Line_Data = []
            Final_Data = []
            # Read line and splitted data into a single dimensional data
            Line_Data = line.split(',')

            #Strips any \n that a data may contain due to csv file system
            count: int = 0
            for idx in range(len(Line_Data)):
                Line_Data[idx] = Line_Data[idx].replace("\n", "")


            # Append a list to a 2-Dimensional Data list
            self.Data_2D.append(Line_Data)

        self.data_rowsize = len(self.Data_2D)
        self.data_colsize = len(self.Data_2D[0])

    def show_dataset_raw(self):
        print("Data row size : ", self.data_rowsize)
        print("Data col size : ", self.data_colsize)


        for idx in range(self.data_rowsize):
            for col in range(len(self.Data_2D[0])):
                print(self.Data_2D[idx] , "\t")



