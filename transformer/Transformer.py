# -*- coding: utf-8 -*-
import torch
import numpy as np
from Layers import *
from Blocks import *


class Transformer(torch.nn.Module):
    """ Defines the transformer model incorporating a user-defined number
        of encoder and decoder blocks.

        int vocab_size:      Number of words in the vocabulary.
        int embedding_size:  Size of the input word vectors.
        int max_seq_length:  Maximum number of words in a sequence.
        int n_encoders:      Number of encoders in encoder stack (default: 2).
        int n_decoders:      Number of encoders in encoder stack (default: 2).
        int attention_size:  Size of the key/query vectors used in the attention
                             calculation (default: 64).
        int attention_heads: Number of self-attention heads to use (default: 8).
    """
    def __init__(self, vocab_size, embedding_size, max_seq_length, n_encoders=2,
                 n_decoders=2, attention_size=128, attention_heads=8, lr=1e-3):
        super(Transformer, self).__init__()
        # Define layers and blocks of Transformer architecture.
        self.embedding = PositionalEmbedding(vocab_size, embedding_size, max_seq_length)

        # Define encoder stack.
        self.encoders = []
        for i in range(n_encoders):
            encoder = EncoderBlock(embedding_size, attention_size, attention_heads)
            self.encoders.append(encoder)

        # Define decoder stack.
        self.decoders = []
        for i in range(n_decoders):
            decoder = DecoderBlock(embedding_size, attention_size, attention_heads)
            self.decoders.append(decoder)

        # Define linear prediction head.
        self.decoder_head = DecoderHead(embedding_size, vocab_size)

        # Initialize Adam optimizer and NLL loss function.
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.training_iter = 0


    def forward(self, enc_x, dec_x):
        """ Computes forward pas from input to output predictions.

            int enc_x:      Tensor of shape (seq_length,) populated with token
                            indices of encoder sequence.
            int dec_x:      Tensor of shape (seq_length,) populated with token
                            indices of decoder sequence.

            returns:        Output tensor of shape (seq_length, vocab_size).
        """
        # Convert integer inputs to word embeddings.
        enc_x = self.embedding(enc_x)
        dec_x = self.embedding(dec_x)

        # Compute encoder outputs.
        for encoder in self.encoders:
            enc_x = encoder(enc_x)
        enc_y = enc_x

        # Compute decoder outputs.
        for decoder in self.decoders:
            dec_x = decoder(dec_x, enc_y)
        return self.decoder_head(dec_x)


    def fit(self, enc_x, dec_x, dec_y, verbosity=10):
        """ Performs single training iteration over a single sample.

            int enc_x:      Tensor of shape (seq_length,) populated with token
                            indices of encoder sequence.
            int dec_x:      Tensor of shape (seq_length,) populated with token
                            indices of decoder sequence (starting with <BOS>).
            int dec_y:      Tensor of shape (seq_length,) populated with token
                            indices of decoder sequence (ending with <EOS>).
            int verbosity:  Specifies after which amount of iterations the loss
                            should be displayed to the user.

            returns:        Output tensor of shape (seq_length, vocab_size).
        """
        # Make a prediction with shape (seq_length, vocab_size).
        pred_y = self(enc_x, dec_x)
        
        # Reshape prediction and target into (1, seq_length, vocab_size) and (1, seq_length).
        pred_y = torch.reshape(pred_y, (1,) + pred_y.shape)
        dec_y = torch.reshape(dec_y, (1,) + dec_y.shape)

        # Accumulate loss over positions in word sequence.
        loss = self.criterion(pred_y[:, 0], dec_y[:, 0])
        for i in range(1, pred_y.shape[1]):
            loss += self.criterion(pred_y[:, i], dec_y[:, i])

        # Print loss every x iters specified by the verbosity setting.
        if self.training_iter % verbosity == 0:
            print("iter {}: loss = {}".format(self.training_iter, loss.item()))
        self.training_iter += 1

        # Perform optimization.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def predict(self, enc_x, dec_x):
        """ Predicts output (sequence) from encoder input and (partial)
            decoder input.

            int enc_x:      Tensor of shape (seq_length,) populated with token
                            indices of encoder sequence.
            int dec_x:      Tensor of shape (seq_length,) populated with token
                            indices of decoder sequence (starting with <BOS>).

            returns:        Output tensor of shape (seq_length,).
        """
        # Make a prediction with shape (seq_length, vocab_size).
        pred_y = self(enc_x, dec_x)
        return torch.argmax(pred_y, dim=1)
