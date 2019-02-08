import torch
import torch.nn as nn


class ATAE_LSTM(nn.Module):

	'''
	This class implements the ATAE_LSTM model
	'''

	def __init__(self, embedding_dimension: int, bidirectional: bool = False, rnn_layers: int = 1, hidden_size: int = 128, rnn_type: str = 'GRU'):

		super(ATAE_LSTM, self).__init__()
		self.embedding_dimension: int = embedding_dimension
		self.bidirectional: bool = bidirectional
		self.rnn_layers: bool = rnn_layers
		self.rnn_type :str = rnn_type
		
		if self.bidirectional:
			self.unprojected_hidden_size = hidden_size * 2
		else:
			self.unprojected_hidden_size = hidden_size	
		
		self.projected_hidden_size = self.projected_hidden_size

		if self.rnn_type == 'GRU':
			self.rnn = torch.nn.GRU(embedding_dimension, self.hidden_size, bidirectional = self.bidirectional, num_layers = self.rnn_layers)
		else:
			self.rnn = torch.nn.LSTM(embedding_dimension, self.hidden_size, bidirectional = self.bidirectional, num_layers = self.rnn_layers)

		self.project_hidden_state = nn.Linear(self.unprojected_hidden_size, self.projected_hidden_size )

		self.
			

				 

