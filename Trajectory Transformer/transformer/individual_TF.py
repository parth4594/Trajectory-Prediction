import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.decoder import Decoder
from transformer.multihead_attention import MultiHeadAttention
from transformer.positional_encoding import PositionalEncoding
from transformer.pointerwise_feedforward import PointerwiseFeedforward
from transformer.encoder_decoder import EncoderDecoder
from transformer.encoder import Encoder
from transformer.encoder_layer import EncoderLayer
from transformer.decoder_layer import DecoderLayer
from transformer.batch import subsequent_mask
from transformer.img_encoding import CNN_Encoder
from transformer.Pretrained_Encoder import PretrainedCNN
import numpy as np
import scipy.io 
import os

import copy
import math

class IndividualTF(nn.Module):
    def __init__(self, enc_inp_size, dec_inp_size, dec_out_size, N=6, in_channels=3, pretrained=False,
                   d_model=512, d_ff=2048, h=8, dropout=0.1,mean=[0,0],std=[0,0]):
        super(IndividualTF, self).__init__()
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model*2)
        ff = PointerwiseFeedforward(d_model*2, d_ff, dropout)
        if pretrained == False:
            img_encoder = CNN_Encoder(in_channels, d_model)
        else:
            img_encoder = PretrainedCNN(d_model, train_CNN = False)

        le_embd = LinearEmbedding(enc_inp_size, d_model)
        #comb_module = Ensemble(le_embd,img_encoder)
        position = PositionalEncoding(d_model*2, dropout)
        self.mean=np.array(mean) 
        self.std=np.array(std)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model*2, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model*2, c(attn), c(attn),
                                 c(ff), dropout), N),
            Ensemble(le_embd,img_encoder,position),
            nn.Sequential(LinearEmbedding(dec_inp_size,d_model*2), c(position)),
            Generator(d_model*2, dec_out_size))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)





    def forward(self, *input):
        return self.model.generator(self.model(*input))

class LinearEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        return self.proj(x)


class Ensemble(nn.Module):
  def __init__(self, LinearEmbedding, CNN_Encoder, PositionalEncoding):
        super(Ensemble, self).__init__()
        self.LinearEmbedding = LinearEmbedding
        self.CNN_Encoder = CNN_Encoder
        self.PositionalEncoding = PositionalEncoding

  def forward(self, x1, x2):
        x1 = self.LinearEmbedding(x1)
        a,b,c,d,e = x2.shape
        x2 = x2.reshape(a*b,c,d,e)
        x2 = self.CNN_Encoder(x2)
        a,b = x2.shape
        x2 = x2.reshape(int(a/8),8,b)
        x = torch.cat((x1, x2), dim=-1)
        x = self.PositionalEncoding(x)
        return x

class multiSequential(nn.Sequential):
  def forward(self, *input):
        for module in self._modules.values():
          input = module(*input)
        return input     
   
 

