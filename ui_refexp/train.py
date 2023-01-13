import argparse
from transformers import DonutProcessor, VisionEncoderDecoderModel
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
from typing import List
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig
from transformers import VisionEncoderDecoderConfig
from PIL import Image, ImageDraw
import json
import math
from datasets import load_dataset
from ui_refexp.donut_dataset import DonutDataset
from ui_refexp.donut_model import DonutModelPLModule
import pytorch_lightning as pl
import os

REFEXP_DATASET_NAME = "ivelin/ui_refexp_saved"
# Pick which pretrained checkpoint to start the fine tuning process from
REFEXP_MODEL_CHECKPOINT = "ivelin/donut-refexp-draft-precision2decs"
# REFEXP_MODEL_CHECKPOINT = "ivelin/donut-refexp-draft"
# REFEXP_MODEL_CHECKPOINT = "naver-clova-ix/donut-base"
# REFEXP_MODEL_CHECKPOINT = "ivelin/donut-docvqa-demo"

# New checkpoint save location after training.
# May be different from the originally loaded checkpoint repo
# if this is a different type of training experiement
SAVE_NEW_CHECKPOINT_REPO = "ivelin/donut-refexp-draft-precision2decs"
# Repo to keep a backup in case the primary gets currupted during save.
BACKUP_REPO_NAME = "ivelin/donut-refexp-draft-precision2decs-backup"

# Normalized image size for encoder pixel input
ENCODER_IMAGE_SIZE = [1280, 960]
# Max sequence for decoder in number of tokens, including prompt and answer.
DECODER_MAX_SEQ_LENGTH = 128


def show_preprocessed_sample(sample):
    """Show info about a dataset sample"""
    image = sample['image']
    width, height = image.size
    print(f"image width, height: {width, height}")
    print(f"prompt: {sample['prompt']}")
    bb = json.loads(sample["target_bounding_box"])
    print(f"target bounding box: {bb}")
    xmin = math.floor(width*bb["xmin"])
    ymin = math.floor(height*bb["ymin"])
    xmax = math.floor(width*bb["xmax"])
    ymax = math.floor(height*bb["ymax"])
    print(
        f"to image pixel values: xmin, ymin, xmax, ymax: {xmin, ymin, xmax, ymax}")


def show_processed_train_sample(processor=None, sample=None):
    pixel_values, decoder_input_ids, labels = sample
    print(pixel_values.shape)
    for decoder_input_id, label in zip(decoder_input_ids.tolist()[:-1], labels.tolist()[1:]):
        if label != -100:
            print(processor.decode([decoder_input_id]),
                  processor.decode([label]))
        else:
            print(processor.decode([decoder_input_id]), label)


def show_processed_val_sample(sample):
    pixel_values, decoder_input_ids, prompt_end_index, processed_parse = val_dataset[0]
    print(pixel_values.shape)
    print(prompt_end_index)
    print(processed_parse)


def load_model():
    pretrained_repo_name = REFEXP_MODEL_CHECKPOINT
    max_length = DECODER_MAX_SEQ_LENGTH
    image_size = ENCODER_IMAGE_SIZE
    # update image_size of the encoder
    # during pre-training, a larger image size was used
    config = VisionEncoderDecoderConfig.from_pretrained(pretrained_repo_name)
    config.encoder.image_size = image_size  # (height, width)
    # update max_length of the decoder (for generation)
    config.decoder.max_length = max_length

    # TODO we should actually update max_position_embeddings and interpolate the pre-trained ones:
    # https://github.com/clovaai/donut/blob/0acc65a85d140852b8d9928565f0f6b2d98dc088/donut/model.py#L602

    processor = DonutProcessor.from_pretrained(pretrained_repo_name)
    model = VisionEncoderDecoderModel.from_pretrained(
        pretrained_repo_name, config=config)
    return (config, processor, model)


def add_tokens(processor=None, list_of_tokens: List[str] = None):
    """
    Add tokens to tokenizer and resize the token embeddings
    """
    newly_added_num = processor.tokenizer.add_tokens(list_of_tokens)
    if newly_added_num > 0:
        model.decoder.resize_token_embeddings(len(processor.tokenizer))


def verify_batch(processor=None, batch=None):
    pixel_values, decoder_input_ids, labels = batch
    print(f"pixel_values.shape: {pixel_values.shape}")
    print(f"decoder_input_ids.shape: {decoder_input_ids.shape}")
    for decoder_input_id, label in zip(decoder_input_ids[0].tolist()[:-1][:50], labels[0].tolist()[1:][:50]):
        if label != -100:
            print(processor.decode([decoder_input_id]),
                  processor.decode([label]))
        else:
            print(processor.decode([decoder_input_id]), label)


def push_to_hub(model_module):
    repo_name = SAVE_CHECKPOINT_REPO

    # here we push the processor and model to the hub
    # note that you can add `private=True` in case you're using the private hub
    # which makes sure the model is only shared with your colleagues
    model_module.processor.push_to_hub(repo_name)
    model_module.model.push_to_hub(repo_name)

    # load back to verify all data was saved OK
    processor = DonutProcessor.from_pretrained(repo_name)
    model = VisionEncoderDecoderModel.from_pretrained(repo_name)

    backup_repo_name = BACKUP_REPO_NAME

    # save a backup in case uploading to the main model fails and corrupts the data
    model_module.processor.push_to_hub(backup_repo_name)
    model_module.model.push_to_hub(backup_repo_name)


def run_training():
    """The main function."""
    dataset = load_dataset(REFEXP_DATASET_NAME)
    print(dataset['train'].info)
    print(dataset)
    # change this index from 0 to split size to see different samples
    sample = dataset['train'][49]
    show_preprocessed_sample(sample)
    # load pre-trained model
    (config, processor, model) = load_model()

    # TODO: Do we need this for UI RefExp? It came from the DocVQA code
    additional_tokens = ["<yes/>", "<no/>"]
    add_tokens(processor=processor, list_of_tokens=additional_tokens)

    # we update some settings which differ from pretraining; namely the size of the images + no rotation required
    # source: https://github.com/clovaai/donut/blob/master/config/train_cord.yaml
    # should be (width, height)
    processor.feature_extractor.size = ENCODER_IMAGE_SIZE[::-1]
    processor.feature_extractor.do_align_long_axis = False

    # For warm up phase, consider picking only a small subset to see if the model converges on the data
    max_train_samples = 500
    # pick a range for sampling
    range_train_samples = range(max_train_samples)

    train_dataset = DonutDataset(REFEXP_DATASET_NAME, max_length=DECODER_MAX_SEQ_LENGTH,  range_samples=range_train_samples,
                                 split="train", task_start_token="<s_refexp>", prompt_end_token="<s_target_bounding_box>",
                                 sort_json_key=False, processor=processor
                                 )

    # pick a small subset for initial val set to see if validation metrics improve
    # max_val_samples = 200
    # range_val_samples = range(max_val_samples)

    val_dataset = DonutDataset(REFEXP_DATASET_NAME, max_length=DECODER_MAX_SEQ_LENGTH,  # range_samples=range_val_samples,
                               split="validation", task_start_token="<s_refexp>", prompt_end_token="<s_target_bounding_box>",
                               sort_json_key=False, processor=processor
                               )

    # show a sample of the processed dataset
    sample = train_dataset[0]
    show_processed_train_sample(processor=processor, sample=sample)

    print(f"train dataset length: {train_dataset.dataset_length}")
    print(f"validation dataset length: {val_dataset.dataset_length}")

    # create corresponding PyTorch dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4)
    # Let's verify a batch:
    batch = next(iter(train_dataloader))
    verify_batch(processor=processor, batch=batch)

    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # clear any previously open wandb logging session
    wandb.finish()

    # initiate PyTorch Lightning module
    config = {"max_epochs": 1,  # 30,
              "val_check_interval": 0.2,  # how many times we want to validate during an epoch
              "check_val_every_n_epoch": 1,
              "gradient_clip_val": 1.0,
              "num_training_samples_per_epoch": 800,
              "lr": 3e-5,
              "train_batch_sizes": [8],
              "val_batch_sizes": [1],
              # "seed":2022,
              "num_nodes": 1,
              "warmup_steps": 300,  # 800/8*30/10, 10%
              "result_path": "./result",
              "verbose": True,
              }
    model_module = DonutModelPLModule(
        config=config, processor=processor, model=model)
    wandb_logger = WandbLogger(project="Donut-RefExp")
    # initialize trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=config.get("max_epochs"),
        val_check_interval=config.get("val_check_interval"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision=16,  # we'll use mixed precision
        num_sanity_val_steps=0,
        logger=wandb_logger,
        # callbacks=[lr_callback, checkpoint_callback],
    )

    # run training
    trainer.fit(model_module, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    # push checkpoint to Hugging Face Hub
    # uncomment when training works on TPU
    # TODO: push_to_hub(model_module)


if __name__ == "__main__":
    # printing environment variables
    print("OS environment variables:")
    for k, v in os.environ.items():
        print(f'{k}={v}')

    # Initialize command line arg parser
    parser = argparse.ArgumentParser(description='Trainer for Donut UI RefExp task.',
                                     prog='Donut UI RefExp Trainer', usage='%(prog)s [options]')

    # Adding optional argument
    parser.add_argument("-a", "--accelerator",  type=str, required=True,
                        help="Set accelerator type for the hardware architecture for the training: cpu, gpu, tpu ")

    # Adding optional argument
    parser.add_argument("-d", "--devices", type=int, required=True,
                        help="Set to number of available devices or cores.")

    # Read arguments from command line
    args = parser.parse_args()

    assert args.accelerator
    print(f"Accelerator set to: {args.accelerator}")
    assert args.devices
    print(f"Devices set to: {args.devices}")

    run_training()
