import torch
import torch.nn as nn
import torch.nn.functional as F 

class Attention(nn.Module):

	'''
	This class implements soft-attention
	'''

	def __init__(self, attention_size: int, batch_first: bool = True):

		self.attention_size: int = attention_size
		self.attention_tensor: torch.Tensor = torch.empty(attention_size, 1)
		torch.nn.init.xavier_normal_(self.attention_tensor.weight)

	def forward(self, attention_candidates: torch.Tensor, weighted_sum_candidates: torch.Tensor = None, batch_first: bool = True) -> torch.Tensor:

		
		if batch_first:
			weights = F.softmax(torch.matmul(attention_candidates, self.attention_tensor), dim=1)
		else:
			dummy = torch.empty(attention_candidates.size(1), attention_candidates.size(0), attention_candidates.size(2))
			
			for i in range(attention_candidates.size(1)):
				dummy[i] = torch.cat([attention_candidates[j, i, :] for j in range(attention_candidates.size(0))],
					 				 dim=0).view(attention_candidates.size(0), attention_candidates.size(2))

			weights = F.softmax(torch.matmul(dummy, self.attention_tensor), dim=1)	

		
		if weighted_sum_candidates is None:
			weighting = weights * attention_candidates
		else:
			weighting = weights * weighted_sum_candidates	

		return torch.sum(weighting, dim=1)




