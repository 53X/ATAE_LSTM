from ATAE_LSTM import ATAE_LSTM
import flair
from flair.data import Sentence

sent1 = Sentence('I am great and python is cool.')


input_sentences = [sent1]
vocab = { 'I':1, 'am':2, 'great':3, 'and':4, 'python':5, 'is':6, 'cool.':7 }


model = ATAE_LSTM()
model.forward(input_sentences=input_sentences, target_words=[[5]], vocab=vocab)