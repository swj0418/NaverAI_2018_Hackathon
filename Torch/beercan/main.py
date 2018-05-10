import sys
import os
from dataloader import Data_Loader

"""
Entry Point
"""

data = Data_Loader(csv_file='beers.csv', root_dir='H:/Datasets/craft-cans')
data.show_dataset_raw()