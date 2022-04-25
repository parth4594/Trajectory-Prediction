# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        #self.img_encoder = img_encoder

    def combination(self,src,src_img):
        comb = self.src_embed(src,src_img)
        return comb

    def forward(self, src, src_img, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """ 
        src = self.src_embed(src,src_img)
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    