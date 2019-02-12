from ATAE_LSTM import ATAE_LSTM
import flair
from flair.data import Sentence
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adagrad
import numpy as np
import pandas as pd
from typing import Union, List
from pathlib import Path



def MyDataset(Dataset):

	def __init__(self, path_to_file: Union[str, Path]):
		
		self.data = pd.read_csv(path_to_file, header='infer')
		self.sentences: List[str] = []
		self.possible_target: List[str] = []
		
		token_list: List[str] = []
		
		
		for sample_no in range(self.data.shape[0]):
			text = self.data.iloc[sample_no]['----------'].lower().strip('\n').strip('\r').replace('\r', '').replace('\n', ' ').strip(' ')
			self.sentences.append(text)
			token_list += text.split(' ')
			
			'''
			This piece of code takes the target word and maps them to the actual word in the sentence 
			and then uses that actual word as the target word rather than using the short hand notation.
			This preprocessing handles the normalization amongst various short hand representations.
			'''
			if self.data.iloc[sample_no]['++++'] == '......':
				self.possible_target.append('**********')
			else:
				for token in text.split(' '):
					if token.find(self.data.iloc[sample_no]['++++']) == 1:
						self.possible_target.append(token)
						break
		token_list = list(np.unique(token_list)) + list(np.unique(self.possible_target))
		self.vocab = dict(j:i for i, j in enumerate(token_list))
		
	
	def __getitem__(self, index: int):
		
		text: str, label: int, target: str = self.sentences[index],
							  				 self.data.iloc[index]['**********_*****'],
							  				 self.possible_target[index]
		label=1 if label!=4 else 0
		target: List[int] = [self.vocab[target]]
		return Sentence(text), label, target	

	def __len__(self):
		return self.data.shape[0]



model  = ATAE_LSTM()
#model.forward(sent, target, vocab)
optimizer = Adagrad(model.parameters())


