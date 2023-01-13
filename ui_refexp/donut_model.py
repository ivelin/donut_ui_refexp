from pathlib import Path
import re
import numpy as np
import math

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from ui_refexp.iou import get_iou

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

# we will use IoU for RefExp instead of edit_distance, which is better suited for DocVQA
# from nltk import edit_distance


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

    def training_step(self, batch, batch_idx):
        pixel_values, decoder_input_ids, labels = batch

        outputs = self.model(pixel_values,
                             decoder_input_ids=decoder_input_ids[:, :-1],
                             labels=labels[:, 1:])
        loss = outputs.loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def token2bbox(self, seq: str):
        target_bbox = self.processor.token2json(seq)
        # safeguard in case text prediction is missing target bbox
        bbox = target_bbox.get('target_bounding_box')
        if bbox is None:
            print(f"token2bbox seq has no target_bounding_box, seq:{seq}")
            bbox = bbox = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
            return bbox
        # print(f"token2 bounding box json: {bbox}")
        # safeguard in case text prediction is missing some bounding box coordinates
        # or coordinates are not valid numeric values
        try:
            xmin = float(bbox.get("xmin", 0))
        except ValueError:
            xmin = 0
        try:
            ymin = float(bbox.get("ymin", 0))
        except ValueError:
            ymin = 0
        try:
            xmax = float(bbox.get("xmax", 1))
        except ValueError:
            xmax = 1
        try:
            ymax = float(bbox.get("ymax", 1))
        except ValueError:
            ymax = 1
        # replace str with float coords
        bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        # print(f"token2 bounding box float: {bbox}")
        return bbox

        def validation_step(self, batch, batch_idx, dataset_idx=0):
            pixel_values, decoder_input_ids, prompt_end_idxs, answers = batch
            decoder_prompts = pad_sequence(
                [input_id[: end_idx + 1]
                    for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
                batch_first=True,
            )

            outputs = self.model.generate(pixel_values,
                                          decoder_input_ids=decoder_prompts,
                                          max_length=max_length,
                                          early_stopping=True,
                                          pad_token_id=self.processor.tokenizer.pad_token_id,
                                          eos_token_id=self.processor.tokenizer.eos_token_id,
                                          use_cache=True,
                                          num_beams=1,
                                          bad_words_ids=[
                                              [self.processor.tokenizer.unk_token_id]],
                                          return_dict_in_generate=True,)

            predictions = []
            for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
                seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(
                    self.processor.tokenizer.pad_token, "")
                # remove first task start token
                seq = re.sub(r"<.*?>", "", seq, count=1).strip()
                predictions.append(seq)

            scores = list()
            for pred, answer in zip(predictions, answers):
                answer = re.sub(r"<.*?>", "", answer, count=1)
                answer = answer.replace(self.processor.tokenizer.eos_token, "")
                answer_bbox = self.token2bbox(answer)
                pred_bbox = self.token2bbox(pred)
                scores.append(get_iou(pred_bbox, answer_bbox))
                if self.config.get("verbose", False) and len(scores) == 1:
                    print(f"      Prediction: {pred}")
                    print(f"      Prediction: {pred}")
                    print(f"          Answer: {answer}")
                    print(f" Prediction bbox: {pred_bbox}")
                    print(f"     Answer bbox: {answer_bbox}")
                    print(f"IoU (bbox match): {scores[0]}")
            return scores

        def validation_epoch_end(self, validation_step_outputs):
            # I set this to 1 manually
            # (previously set to len(self.config.dataset_name_or_paths))
            num_of_loaders = 1
            if num_of_loaders == 1:
                validation_step_outputs = [validation_step_outputs]
            assert len(validation_step_outputs) == num_of_loaders
            cnt = [0] * num_of_loaders
            total_metric = [0] * num_of_loaders
            val_metric = [0] * num_of_loaders
            for i, results in enumerate(validation_step_outputs):
                for scores in results:
                    cnt[i] += len(scores)
                    total_metric[i] += np.sum(scores)
                val_metric[i] = total_metric[i] / cnt[i]
                val_metric_name = f"val_metric_{i}th_dataset"
                self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
            self.log_dict({"val_metric": np.sum(total_metric) /
                          np.sum(cnt)}, sync_dist=True)

        def configure_optimizers(self):
            # TODO add scheduler
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.config.get("lr"))

            return optimizer

        def train_dataloader(self):
            return train_dataloader

        def val_dataloader(self):
            return val_dataloader
