# -*- coding: utf-8 -*-
import torch
from torch.nn import Embedding
import numpy as np



class PositionalEmbedding(torch.nn.Module):
    """ A pair of lookup tables which provide positional word embeddings obtained,
        by learning separate word embeddings (one per word) and position embeddings
        (one per sequence position) and adding these embeddings together.

        int vocab_size:       Size of the vocabulary.
        int embedding_size:   Size of the word/position vectors.
        int max_seq_length:   Maximum number of words in a sequence.

        Tensor x:             Input tensor of shape (<=max_seq_length,).
        returns:              Output tensor of shape (<=max_seq_length, embedding_size)
    """
    def __init__(self, vocab_size, embedding_size, max_seq_length):
        super(PositionalEmbedding, self).__init__()
        
        # Defines embedding for each word in the vocabulary (and padding token 0).
        self.word_embeddings = Embedding(vocab_size, embedding_size)
        
        # Defines an embedding for every possible position in the sequence.
        self.pos_embeddings = torch.randn(max_seq_length, embedding_size, requires_grad=True)

    def forward(self, x):
        word_embeddings = self.word_embeddings(x)
        pos_embeddings = self.pos_embeddings[:x.shape[0]]
        return word_embeddings.add(pos_embeddings)



class SelfAttention(torch.nn.Module):
    """ Performs (self-)attention on its input.

        int embedding_size: Size of the input word vectors.
        int key_size:       Size of the key/query vectors used in the attention
                            calculation.
        int masked:         Whether to mask future words when computing attention.
                            Used by transformer decoder only (default: False).

        Tensor x:           Input tensor of shape (seq_length, embedding_size) when
                            using self-attention. A list of two of these tensors in
                            case of encoder-decoder attention.
        returns:            Output tensor of shape (seq_length, embedding_size)
    """
    def __init__(self, embedding_size, key_size, masked=False):
        super(SelfAttention, self).__init__()
        # Predefine scaling factor used in attention computation.
        self.sqrt_key_size = torch.sqrt(torch.Tensor([key_size]))
        
        # Define query, key and value transforms and Softmax layer.
        self.Wq = torch.randn(embedding_size, key_size, requires_grad=True)
        self.Wk = torch.randn(embedding_size, key_size, requires_grad=True)
        self.Wv = torch.randn(embedding_size, key_size, requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.masked = masked

    def forward(self, x):
        if isinstance(x, list):
            # If x contains multiple inputs (i.e. encoder outputs and
            # decoder inputs), then perform encoder-decoder attention.
            dec_x, enc_y = x
            Q = torch.mm(dec_x, self.Wq)
            K = torch.mm(enc_y, self.Wk)
            V = torch.mm(enc_y, self.Wv)
            x = dec_x
        else:
            # Otherwise perform regular self-attention on input.
            Q = torch.mm(x, self.Wq)
            K = torch.mm(x, self.Wk)
            V = torch.mm(x, self.Wv)

        # Compute dot-product between queries and keys.
        QK = torch.mm(Q, K.t()) / self.sqrt_key_size
        
        # Mask out future words if masking is enabled.
        if self.masked and QK.shape[0] == QK.shape[1]:
            mask = 1 - torch.triu(torch.ones(QK.shape[0], QK.shape[1]), diagonal=1)
            QK = mask * QK + (1 - mask) * -1e32
        
        # Weights values V by similarity between keys and queries.
        return torch.mm(self.softmax(QK), V)



class MultiHeadAttention(torch.nn.Module):
    """ Performs attention using multiple self-attention heads, whose outputs are subsequently
        concatenated as a matrix of shape (seq_length, heads*key_size) and reduced to an
        output shape of (seq_length, embedding_size).

        int embedding_size: Size of the input word vectors.
        int key_size:       Size of the key/query vectors used in the attention
                            calculation.
        int heads:          Number of self-attention heads to use (default: 8).
        int masked:         Whether to mask future words when computing attention.
                            Used by transformer decoder only (default: False).

        Tensor x:           Input tensor of shape (seq_length, embedding_size) when
                            using self-attention. A list of two of these tensors in
                            case of encoder-decoder attention.
        returns:            Output tensor of shape (seq_length, embedding_size).
    """
    def __init__(self, embedding_size, key_size, heads=8, masked=False):
        super(MultiHeadAttention, self).__init__()
        # Create multiple self-attention heads.
        self.heads = [SelfAttention(embedding_size, key_size, masked) for i in range(heads)]

        # Define matrix to reduce dimensionality of concatenated attention matrix.
        self.W = torch.randn(heads * key_size, embedding_size, requires_grad=True)

    def forward(self, x):
        # Concatenate head outputs to matrix of shape (tokens, heads*embedding_dim).
        concat_attention = torch.cat([head(x) for head in self.heads], dim=1)
        return torch.mm(concat_attention, self.W)




class DecoderHead(torch.nn.Module):
    """ Takes decoder output of shape (seq_length, embedding_size) and performs a
        feed-forward network on each word, followed by a log soxtmax activation
        to obtain word likelihood predictions for each word in the vocabulary.

        int embedding_size: Size of the input word vectors.
        int vocab_size:     Size of the vocabulary.

        Tensor x:           Input tensor(s) of shape (seq_length, embedding_size).
        returns:            Output tensor of shape (seq_length, vocab_size)
    """
    def __init__(self, embedding_size, vocab_size):
        super(DecoderHead, self).__init__()
        self.linear = torch.nn.Linear(embedding_size, vocab_size)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        return self.logsoftmax(self.linear(x))

