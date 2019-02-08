import torch
import torch.nn as nn
from .attention import Attention
import torch.nn.functional as F 


class ATAE_LSTM(nn.Module):

	'''
	This class implements the ATAE_LSTM model
	'''

	def __init__(self, embedding_dimension: int, bidirectional: bool = False, rnn_layers: int = 1,
				 hidden_size: int = 128, rnn_type: str = 'GRU', aspect_embedding_dimension: int):

		super(ATAE_LSTM, self).__init__()
		self.embedding_dimension: int = embedding_dimension
		self.bidirectional: bool = bidirectional
		self.rnn_layers: bool = rnn_layers
		self.rnn_type :str = rnn_type
		self.aspect_embedding_dimension = aspect_embedding_dimension
		
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

		self. attention = Attention(attention_size = )

		self.aspect_projecting_layer = nn.Linear(self.aspect_size, self.new_aspect_size)


		def weird_operation(self, rnn_output_tensor: torch.Tensor, aspect_embedding_tensor: torch.Tensor) -> torch.Tensor:

			transformed_rnn_output = self.project_hidden_state(rnn_output_tensor.size(-1), rnn_output_tensor.size(-1))
			
			concat_aspect_tensor = torch.cat([transformed_aspect_embedding for i in range(transformed_rnn_output.size(1))], dim=1)
			return F.relu(torch.cat([transformed_rnn_output, concat_aspect_tensor], dim = 2))

		
		def custom_embedding_layer(self):

			pass

		def aspect_embedding_layer(self, weights: torch.Tensor, 
								   trainable: bool = True, affine: bool = False,
								   input_indexes: List[List[int]], vocab_size: int, 
								   embedding_dim: int = 300) -> torch.Tensor:

			embedding = nn.Embedding(vocab_size, embedding_dim)(input_indexes)
			embedding.from_pretrained(weights, freeze = trainable)
			em
			if affine:
				transformed_aspect_embeddings = self.aspect_projecting_layer(embedding)
			else:
				transformed_aspect_embeddings = embedding

			return transformed_aspect_embeddings





			

				 

