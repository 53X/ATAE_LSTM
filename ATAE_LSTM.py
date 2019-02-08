import torch
import torch.nn as nn
from .attention import Attention
import torch.nn.functional as F 
import flair
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, WordEmbeddings


class ATAE_LSTM(nn.Module):

	'''
	This class implements the ATAE_LSTM model
	'''

	def __init__(self, embedding_dimension: int, bidirectional: bool = False, rnn_layers: int = 1,
				 hidden_size: int = 128, rnn_type: str = 'GRU', aspect_embedding_dimension: int,
				 attention_size: int):

		super(ATAE_LSTM, self).__init__()
		self.embedding_dimension: int = embedding_dimension
		self.bidirectional: bool = bidirectional
		self.rnn_layers: bool = rnn_layers
		self.rnn_type :str = rnn_type
		# self.aspect_embedding_dimension = aspect_embedding_dimension
		
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
		self. attention = Attention(attention_size = attention_size)
		self.aspect_projecting_layer = nn.Linear(self.aspect_size, self.new_aspect_size)
		
		self.stackedembeddings: StackedEmbeddings = StackedEmbeddings([FlairEmbeddings('news-forward'), 
																	   FlairEmbeddings('news-backward')])

				

		def weird_operation(self, rnn_output_tensor: torch.Tensor, aspect_embedding_tensor: torch.Tensor) -> torch.Tensor:

			transformed_rnn_output = self.project_hidden_state(rnn_output_tensor.size(-1), rnn_output_tensor.size(-1))(rnn_output_tensor)
			
			concat_aspect_tensor = torch.cat([transformed_aspect_embedding for i in range(transformed_rnn_output.size(1))], dim=1)
			return F.relu(torch.cat([transformed_rnn_output, concat_aspect_tensor], dim = 2))

		
		

		def custom_embedding_layer(self, inputs: Union[List[Sentence], Sentence]) -> torch.Tensor:

			if type(inputs) == Sentence:
				inputs = [inputs]

			self.stackedembeddings.embed(inputs)
			inputs.sort(lambda x: len(x), reverse = True)
			max_length = len(inputs[0])
			batch_tensor_list : List[torch.Tensor] = []
			for sentence in inputs:
				sentence_tensor = [token.get_embedding().unsqueeze(dim=0) for token in sentence.tokens]
				for i in range(max_length-len(sentence)):
					sentence_tensor.append(torch.zeros(token.get_embedding().size(0)).unsqueeze(0))
					sentence_tensor = torch.cat(sentence_tensor, dim=0)
				batch_tensor_list.append(sentence_tensor.unsqueeze(0))
			batch_tensor: torch.Tensor = torch.cat(batch_tensor_list, dim=0)

			return batch_tensor					



		def aspect_embedding_layer(self, trainable: bool = True, affine: bool = False, vocab_size: int,
								   targets:Tuple(List[List[int]], List[(Sentence)]), embedding_dim: int = 300) -> torch.Tensor:

			self.stackedembeddings.embed(targets[1])
			embed = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx = 0)
			weights = torch.empty(vocab_size + 1, embedding_dim)
			for i, sentence in targets[1]:
				for j, token in enumerate(sentence.tokens):
					index = targets[0][i][j]
					weights[i] = token.get_embedding()

			embed.from_pretrained(weights = weights, freeze = False)
			inputs = torch.LongTensor(targets[0])
			embedding = embed(inputs)

			if affine:
				transformed_aspect_embeddings = self.aspect_projecting_layer(embedding)
			else:
				transformed_aspect_embeddings = embedding

			return transformed_aspect_embeddings

		
		def affine_transformation_final(sent_repr: torch.tensor, final_hidden_state: torch.Tensor, projections: List[int]):

			projected_final_hidden = self.projected_hidden_size(final_hidden_state.size(-1), final_hidden_state.size(-1))(final_hidden_state)
			projected_sent_repr= self.project_hidden_state(sent_repr.size(-1), sent_repr.size(-1))(sent_repr)
			
			return self.project_hidden_state(F.tanh(projected_final_hidden + projected_sent_repr).size(-1), 2)

		def forward(self, )





			

				 

