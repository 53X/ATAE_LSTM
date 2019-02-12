from ATAE_LSTM import ATAE_LSTM
import flair
from flair.data import Sentence
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adagrad

import pandas as pd
from typing import Union
from pathlib import Path



def MyDataset(Dataset):

	def __init__(self, path_to_file: Union[str, Path]):

		self.data = pd.read_csv(path_to_file, header='infer')
		
	def __getitem__(self, index: int):
		sample = self.data.iloc[index]




	def __len__(self):
		pass


sent = [Sentence('I am good')]
vocab = {'I':1, 'am':2, 'good':3}
target = [[1]]

model  = ATAE_LSTM()
model.forward(sent, target, vocab)
#optimizer = Adagrad(model.parameters())


