# -*- coding: utf-8 -*-
import torch
import numpy as np
from Layers import *


class EncoderBlock(torch.nn.Module):
    """ Defines the transformer encoder block using one multi-head attention layer
        and a feed forward, linear layer. Additional residual connections are used
        to allow info to flow past multi-head attention and linear layers.

        int embedding_size:  Size of the input word vectors.
        int attention_size:  Size of the key/query vectors used in the attention
                             calculation.
        int attention_heads: Number of self-attention heads to use (default: 8).

        Tensor x:           Input tensor of shape (seq_length, embedding_size).
        returns:            Output tensor of shape (seq_length, embedding_size).
    """
    def __init__(self, embedding_size, attention_size, attention_heads=8):
        super(EncoderBlock, self).__init__()
        # Define layers of encoder block.
        self.attention = MultiHeadAttention(embedding_size, attention_size, attention_heads, masked=False)
        self.addnorm = torch.nn.LayerNorm(embedding_size)
        self.feedforward = torch.nn.Linear(embedding_size, embedding_size)
                
    def forward(self, x):
        x2 = self.attention(x)
        x3 = self.addnorm(x2.add(x)) # Residual
        x4 = self.feedforward(x3)
        return self.addnorm(x4.add(x3)) # Residual



class DecoderBlock(torch.nn.Module):
    """ Defines the transformer decoder block using one masked multi-head attention
        layer and a feed forward, linear layer. Additional residual connections are
        used to allow info to flow past multi-head attention and linear layers.

        int embedding_size:  Size of the input word vectors.
        int attention_size:  Size of the key/query vectors used in the attention
                             calculation.
        int attention_heads: Number of self-attention heads to use (default: 8).

        Tensor x:           Input tensor of shape (seq_length, embedding_size).
        returns:            Output tensor of shape (seq_length, embedding_size).
    """
    def __init__(self, embedding_size, attention_size, attention_heads=8):
        super(DecoderBlock, self).__init__()
        # Define masked multi-head attention as first layer of decoder block.
        self.attention = MultiHeadAttention(embedding_size, attention_size, attention_heads, masked=True)
        self.addnorm = torch.nn.LayerNorm(embedding_size)

        # Perform encoder-decoder attention instead of regular self-attention.
        self.enc_dec_attention = MultiHeadAttention(embedding_size, attention_size, attention_heads, masked=False)
        
        self.feedforward = torch.nn.Linear(embedding_size, embedding_size)
        
    def forward(self, dec_x, enc_y):
        x2 = self.attention(dec_x)
        x3 = self.addnorm(x2.add(dec_x)) # Residual

        x4 = self.enc_dec_attention([x3, enc_y])
        x5 = self.addnorm(x4.add(x3)) # Residual
        
        x6 = self.feedforward(x5)
        return self.addnorm(x6.add(x5)) # Residual

