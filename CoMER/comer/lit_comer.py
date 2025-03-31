import zipfile
from typing import List
import editdistance
import json

import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor

from comer.datamodule import Batch, vocab
from comer.model.comer import CoMER
from comer.utils.utils import (ExpRateRecorder, Hypothesis, ce_loss,focal_loss_with_ignore,
                               to_bi_tgt_out)
from comer.utils.generation_pred import generate_default_label_set, generate_new_label_set, out_hat_to_pred


class LitCoMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        #vocab_size: int,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        # training
        learning_rate: float,
        patience: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.comer_model = CoMER(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            #vocab_size=vocab_size,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

        self.exprate_recorder = ExpRateRecorder()

    def forward(
        self, img: FloatTensor, img_mask: LongTensor,text: LongTensor, text_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.comer_model(img, img_mask, text, text_mask, tgt)

    def training_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)

        pred_paddded, pred_mask, error_mask = generate_default_label_set(len(batch.indices), self.device)
        
        ogl_loss=0

        if self.current_epoch > 300:
            out_hat = self(batch.imgs, batch.mask, pred_paddded,pred_mask, tgt)
            pred = out_hat_to_pred(out_hat, 5)
            ogl_loss = ce_loss(out_hat, out)
            pred_paddded, pred_mask, error_mask = generate_new_label_set(pred, batch.indices, self.device)

        
        out_hat = self(batch.imgs, batch.mask, pred_paddded, pred_mask, tgt)

        epx_loss = ce_loss(out_hat, out)
        #msk_loss = focal_loss_with_ignore(msk_hat, error_mask, ignore_value=-1)
        loss = epx_loss + ogl_loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        #self.log("msk_loss", msk_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("epx_loss", epx_loss, on_step=False, on_epoch=True, sync_dist=True)
        #self.log("ogl_loss", ogl_loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Batch, _):
        pred_paddded, pred_mask, error_mask = generate_default_label_set(len(batch.indices), self.device)

        hyps= self.approximate_joint_search(batch.imgs, batch.mask, pred_paddded, pred_mask)
        pred_paddded, pred_mask, error_mask = generate_new_label_set([h.seq for h in hyps], batch.indices, self.device)
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat  = self(batch.imgs, batch.mask, pred_paddded, pred_mask, tgt)
        epx_loss = ce_loss(out_hat, out)
        msk_loss = focal_loss_with_ignore(msk_hat, error_mask, ignore_value=-1)
        loss = epx_loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        hyps= self.approximate_joint_search(batch.imgs, batch.mask, pred_paddded, pred_mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

    def test_step(self, batch: Batch, _):
        pred_paddded, pred_mask, error_mask = generate_default_label_set(len(batch.indices), self.device)

        hyps,error_mask = self.approximate_joint_search(batch.imgs, batch.mask, pred_paddded, pred_mask)
        pred_paddded, pred_mask, error_mask = generate_new_label_set([h.seq for h in hyps], batch.indices, self.device)
        hyps, error_mask= self.approximate_joint_search(batch.imgs, batch.mask, pred_paddded, pred_mask)


        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        gts = [vocab.indices2words(ind) for ind in batch.indices]
        preds = [vocab.indices2words(h.seq) for h in hyps]

        error_mask = error_mask.cpu().numpy().tolist()
        old_pred = pred_paddded.cpu().numpy().tolist()
        old_pred = [vocab.indices2words(p) for p in old_pred]
        return batch.img_bases, preds,gts, error_mask, old_pred

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        errors_dict = {}
        predictions_dict = {}
        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds, gts, error_mask, old_pred in test_outputs:
                for img_base, pred, gt,error_mask, old_pred in zip(img_bases, preds, gts,error_mask, old_pred):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)
                    distance = editdistance.eval(pred, gt)
                    if distance > 0:
                        errors_dict[img_base] = {
                            "pred": " ".join(pred),
                            "gt": " ".join(gt),
                            "dist": distance,
                            "error_mask": error_mask,
                            "old_pred": old_pred,
                        }

                    predictions_dict[img_base] = {
                        "pred": " ".join(pred),
                        "gt": " ".join(gt),
                        "dist": distance,
                        "error_mask": error_mask,
                        "old_pred": old_pred,
                    }
        with open("errors.json", "w") as f:
            json.dump(errors_dict, f)
        with open("predictions.json", "w") as f:
            json.dump(predictions_dict, f)

    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor,text: LongTensor, text_mask: LongTensor
    ) -> List[Hypothesis]:
        return self.comer_model.beam_search(img, mask, text,text_mask,**self.hparams)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

        reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.25,
            patience=self.hparams.patience // self.trainer.check_val_every_n_epoch,
        )
        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": "val_ExpRate",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
