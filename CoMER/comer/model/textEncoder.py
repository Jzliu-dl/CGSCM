import torch.nn as nn
import pytorch_lightning as L
from comer.model.pos_enc import WordPosEnc
from torch import FloatTensor, LongTensor
from einops import rearrange


class TextEncoder(L.LightningModule):
    def __init__(
        self, 
        d_model: int,
        nhead: int,
        num_encoder_layers:int,
        dim_feedforward: int, 
        dropout: float, 
        vocab_size: int = 114,
    ):
        super().__init__()
        print('vocab_size:',vocab_size)

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                #batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )

    def forward(
            self,
            src: LongTensor,
            src_mask: LongTensor,
    ) -> FloatTensor:
        """encode src to feature

        Parameters
        ----------
        src : LongTensor
            [b, l]
        src_mask : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, d]
        """
        # embed
        src = self.word_embed(src)
        src = src + self.pos_enc(src)

        src = rearrange(src,'b l d -> l b d')

        # transformer encoder
        src = self.model(src, src_key_padding_mask=src_mask)

        src = rearrange(src,'l b d -> b l d')

        return src
