# -*- coding: utf-8 -*-
import torch
import numpy as np
from Transformer import Transformer

# Create random Tensors to hold inputs and outputs
encoder_input = torch.randint(0, 100, (32,))
decoder_input = torch.tensor(np.arange(0, 15) + 1)
decoder_output = torch.tensor(np.arange(0, 15) + 2)
    

# Construct our model by instantiating the class defined above
model = Transformer(vocab_size=100,
                    embedding_size=256,
                    max_seq_length=32)

# Perform 100 training iterations.
for i in range(40):
    model.fit(encoder_input, decoder_input, decoder_output)

# Compare labels and predictions.
print("\nenc inputs:\n", encoder_input)
print("\ndec inputs:\n", decoder_input)
print("\ndec labels:\n", decoder_output)
print("\ndec prediction:\n", model.predict(encoder_input, decoder_input))
