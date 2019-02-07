import torch
import torch.nn as nn
import torch.nn.functional as F 

class Attention(nn.Module):

	'''
	This class implements soft-attention
	'''

	def __init__(self, attention_size: int):

		self.attention_size: int = attention_size
		self.attention_tensor: torch.Tensor = torch.empty(attention_size, 1)
		torch.nn.init.xavier_normal_(self.attention_tensor.weight)

	def forward(self, attention_candidates: torch.Tensor, weighted_sum_candidates: torch.Tensor = None) -> torch.Tensor:

		weights = F.softmax(torch.matmul(attention_candidates, self.attention_tensor), dim=1)
		
		if weighted_sum_candidates is None:
			weighting = weights * attention_candidates
		else:
			weighting = weights * weighted_sum_candidates	

		return torch.sum(weighting, dim=1)


class ATAE_LSTM(nn.module):

	'''
	The ATAE_LSTM model
	'''

	def __init__(self, embeddings_dimension:int, )


