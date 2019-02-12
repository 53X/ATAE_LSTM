import torch
import torch.nn as nn
from attention import Attention
import torch.nn.functional as F 
import flair
from flair.data import Sentence
from flair.embeddings import StackedEmbeddings, FlairEmbeddings, WordEmbeddings
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import Union, List, Tuple


class ATAE_LSTM(nn.Module):

    '''
    This class implements the ATAE_LSTM model
    '''

    def __init__(self, num_classes: int = 2, bidirectional: bool = False, rnn_layers: int = 1,
                 hidden_size: int = 256, rnn_type: str = 'GRU'):

        super(ATAE_LSTM, self).__init__()

        self.stackedembeddings: StackedEmbeddings = StackedEmbeddings([FlairEmbeddings('news-forward'), 
                                                                       FlairEmbeddings('news-backward')])
        self.wordembeddings: StackedEmbeddings = StackedEmbeddings([WordEmbeddings('glove')])
        self.embedding_dimension: int = self.stackedembeddings.embedding_length + self.wordembeddings.embedding_length
        self.bidirectional: bool = bidirectional
        self.rnn_layers: int = rnn_layers
        self.rnn_type :str = rnn_type
        self.num_classes: int = num_classes
        self.hidden_size: int = hidden_size
        
        
        if self.rnn_type == 'GRU':
            self.rnn = torch.nn.GRU(self.embedding_dimension, self.hidden_size, bidirectional = self.bidirectional, num_layers = self.rnn_layers)
        else:
            self.rnn = torch.nn.LSTM(self.embedding_dimension, self.hidden_size, bidirectional = self.bidirectional, num_layers = self.rnn_layers)

        self.attention = Attention()
 
    def weird_operation(self, rnn_output_tensor: torch.tensor, aspect_embedding_tensor: torch.tensor) -> torch.tensor:

            transformed_rnn_output = self.myLinear(rnn_output_tensor)
            return F.relu(torch.cat([transformed_rnn_output, aspect_embedding_tensor], dim = 2))

    def myLinear(self, input_tensor: torch.tensor, output_dim: int = None, keep=True) -> torch.tensor:
        
        if keep :
            output_dim = input_tensor.size(-1)
        elif output_dim is None:
            raise Exception('Enter the output dimension')
        
        self.Linear = nn.Linear(input_tensor.size(-1), output_dim)
        torch.nn.init.xavier_uniform(self.Linear.weight)
        return self.Linear(input_tensor)




    def custom_embedding_layer(self, inputs: Union[List[Sentence], Sentence]) -> torch.tensor:

        if type(inputs) == Sentence:
            inputs = [inputs]

        self.stackedembeddings.embed(inputs)
        inputs.sort(key=lambda x: len(x), reverse = True)
        max_length = len(inputs[0])
        lengths: List[int] = []
        batch_tensor_list : List[torch.Tensor] = []
        for sentence in inputs:
            sentence_tensor = [token.get_embedding().unsqueeze(dim=0) for token in sentence.tokens]
            for i in range(max_length-len(sentence)):
                sentence_tensor.append(torch.zeros(token.get_embedding().size(0)).unsqueeze(0))
            sentence_tensor = torch.cat(sentence_tensor, dim=0)
            batch_tensor_list.append(sentence_tensor.unsqueeze(0))
            lengths.append(len(sentence))
        batch_tensor: torch.Tensor = torch.cat(batch_tensor_list, dim=0)

        return batch_tensor, torch.tensor(lengths)



    def aspect_embedding_layer(self, targets: tuple([List[List[int]], List[Sentence]]),  vocab: dict(),
                               trainable: bool = True, affine: bool = False) -> torch.tensor:

        self.wordembeddings.embed(targets[1])
        embed = nn.Embedding(len(vocab), self.wordembeddings.embedding_length, padding_idx = 0)
        weights = torch.empty(len(vocab), self.wordembeddings.embedding_length)
        
        '''
        for i, sentence in targets[1]:
            for j, token in enumerate(sentence.tokens):
                index = targets[0][i][j]
                weights[i] = token.get_embedding()
        '''
        embed.from_pretrained(embeddings = weights, freeze = not trainable)
        inputs = torch.tensor(targets[0])
        embedding = embed(inputs)
        if affine:
            transformed_aspect_embeddings = self.myLinear(input_tensor=embedding, keepdim=True)
        else:
            transformed_aspect_embeddings = embedding
        return transformed_aspect_embeddings


    def affine_transformation_final(self, sent_repr: torch.tensor, final_hidden_state: torch.tensor):

        projected_final_hidden = self.myLinear(final_hidden_state, output_dim=sent_repr.size(-1), keep=False)
        projected_sent_repr= self.myLinear(sent_repr, keep=True)
        print (self.myLinear(input_tensor=torch.tanh(projected_sent_repr + projected_final_hidden), output_dim=self.num_classes, keep=False))

    
    def forward(self, input_sentences: List[Sentence], target_words: List[List[int]], vocab: dict()):

        
        nontrainable_embeddings, __lengths__ = self.custom_embedding_layer(inputs = input_sentences)
        trainable_embeddings = self.aspect_embedding_layer(targets = tuple([target_words, input_sentences]), vocab = vocab)
        trainable_embeddings = torch.cat([trainable_embeddings for i in range(nontrainable_embeddings.size(1))], dim=1)
        combined_embeddings = torch.cat([nontrainable_embeddings, trainable_embeddings], dim = 2)
        
        packed_embeddings = pack_padded_sequence(combined_embeddings, lengths =__lengths__, batch_first=True)
        recurrent_output, last_states = self.rnn(packed_embeddings)
        padded_rnn_embedding, __ = pad_packed_sequence(recurrent_output, batch_first=True)
        
        weird_tensor = self.weird_operation(rnn_output_tensor = padded_rnn_embedding, aspect_embedding_tensor = trainable_embeddings)
        final_hidden_state_repr = torch.cat([padded_rnn_embedding[:, -1, :], padded_rnn_embedding[:, 0, :]], dim=-1).unsqueeze(dim=1)
        sent_repr = self.attention.forward(attention_candidates = weird_tensor, attention_size = weird_tensor.size(-1))

        final_logits = self.affine_transformation_final(sent_repr = sent_repr, final_hidden_state = final_hidden_state_repr)

        return final_logits