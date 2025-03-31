from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor
from einops import rearrange

from comer.utils.utils import Hypothesis

from .decoder import Decoder
from .visEncoder import VisEncoder
from .textEncoder import TextEncoder
from .mask import maskPredictor


class CoMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        dc: int,
        #vocab_size: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()
        vocab_size = 114

        self.visEncoder = VisEncoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.textEncoder = TextEncoder(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            vocab_size=vocab_size
        )
        self.maskPredictor = maskPredictor(d_model=d_model)

        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, text: LongTensor, text_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        text : LongTensor
            [b, l]
        text_mask: LongTensor
            [b, l]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        img_feature, img_mask  = self.visEncoder(img, img_mask)  # [b, h, w, d]
        text_feature = self.textEncoder(text, text_mask) # [b, t_text, d]
        
        error_mask = self.maskPredictor(img_feature, text_feature, img_mask, text_mask)
        text_feature = text_feature * error_mask.unsqueeze(-1)
        out = self.decoder(img_feature, img_mask, text_feature, text_mask, tgt)
        return out,error_mask

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        text: LongTensor,
        text_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        text : LongTensor
            [b, l]
        text_mask: LongTensor
            [b, l]
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.visEncoder(img, img_mask) # [b, h, w, d]
        
        text_feature = self.textEncoder(text, text_mask) # [b, t_text, d]
        error_mask = self.maskPredictor(feature, text_feature, mask, text_mask)

        text_feature = text_feature * error_mask.unsqueeze(-1)

        return self.decoder.beam_search(
            [feature], [mask], [text_feature], [text_mask], beam_size, max_len, alpha, early_stopping, temperature
        ),error_mask
