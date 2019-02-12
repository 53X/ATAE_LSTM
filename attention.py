import torch
import torch.nn as nn
import torch.nn.functional as F 

class Attention(nn.Module):

    '''
    This class implements soft-attention
    '''

    def __init__(self, batch_first: bool = True):

        super(Attention, self).__init__()
        self.batch_first: bool = batch_first
        

    def forward(self, attention_candidates: torch.Tensor, attention_size: int, weighted_sum_candidates: torch.Tensor = None) -> torch.Tensor:

        self.attention_tensor: torch.Tensor = torch.empty(attention_candidates.size(0), attention_size, 1) 
        torch.nn.init.ones_(self.attention_tensor)

        if self.batch_first:
            weights = F.softmax(torch.matmul(attention_candidates, self.attention_tensor), dim=1)
        else:
            dummy = torch.empty(attention_candidates.size(1), attention_candidates.size(0), attention_candidates.size(2))
                
            for i in range(attention_candidates.size(1)):
                dummy[i] = torch.cat([attention_candidates[j, i, :] for j in range(attention_candidates.size(0))],
                                     dim=0).view(attention_candidates.size(0), attention_candidates.size(2))

            weights = F.softmax(torch.matmul(dummy, self.attention_tensor), dim=1)  

        #print(weights.size())

        
        if weighted_sum_candidates is None:
            weighting = torch.sum(weights*attention_candidates, dim=1, keepdim=True)
        else:
            weighting = torch.sum(weights*weighted_sum_candidates, dim=1, keepdim=True)  

        return weighting