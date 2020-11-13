# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Code modified by: Siddharth Vashishtha 20 May, 2020

'''
Usage:

python run_finetuning_nli.py \
          --pre_model_path "roberta-large" \
          --train_folder "train_dummy/" \
          --dev_folder "dev_dummy/" \
          --test_folder "test_dummy/" \
          --train_file "train_dummy.tsv" \
          --dev_file "dev_dummy.tsv" \
          --test_file "test_dummy.tsv" \
          --num_labels 2 \
          --hypothesis_only False \
          --train_batch_size 16 \
          --num_train_epochs 1 \
	  --learning_rate 2e-5 \
          --weight_decay 0.1 \
          --warmup_steps 122 \
          --logging_steps 200 \
          --save_steps 500 

'''
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"  #specify which gpus to use

import pickle
## For saving objects to pickles
def save_obj(obj, name ):
    with open(name , 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

import argparse

import dataclasses

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction 
from glue_data_huggingface import GlueDataset
from glue_data_huggingface import GlueDataTrainingArguments as DataTrainingArguments
from glue_processor_huggingface import glue_compute_metrics
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from trainer_huggingface import Trainer 

logging.basicConfig(level=logging.INFO)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

def compute_metrics(p: EvalPrediction) -> Dict:
    '''
    Task specific way of computing metrics
    '''
    preds = np.argmax(p.predictions, axis=1)
    
    return glue_compute_metrics('dnc2', preds, p.label_ids)


def main():

    parser = argparse.ArgumentParser(description='Fine-tune NLI models')
    
    parser.add_argument('--pre_model_path', type=str,
                        default='roberta-large',
                        help='pretrained model name or model path')

    parser.add_argument('--train_folder', type=str,
                        default='',
                        help='')

    parser.add_argument('--dev_folder', type=str,
                        default='',
                        help='')

    parser.add_argument('--test_folder', type=str,
                        default='',
                        help='')

    parser.add_argument('--train_file', type=str,
                        default='',
                        help='')

    parser.add_argument('--dev_file', type=str,
                        default='',
                        help='')

    parser.add_argument('--test_file', type=str,
                        default='',
                        help='')

    parser.add_argument('--num_labels', type=int,
                        default=2,
                        help='num of classification labels in the dataset')

    parser.add_argument('--hypothesis_only', type=str,
                        default='false',
                        help="True if running a hypothesis model only -- all inputs in ['yes', 'true', 't', '1'] assume True else False")

    parser.add_argument('--train_batch_size', type=int,
                        default=32,
                        help='')
    parser.add_argument('--num_train_epochs', type=int,
                        default=3,
                        help='')
    parser.add_argument('--learning_rate', type=float,
                        default=2e-5,
                        help='')
    parser.add_argument('--weight_decay', type=float,
                        default=0.1,
                        help='')
    parser.add_argument('--warmup_steps', type=int,
                        default=0,
                        help='')
    parser.add_argument('--logging_steps', type=int,
                        default=500,
                        help='')
    parser.add_argument('--save_steps', type=int,
                        default=1000,
                        help='')

    args = parser.parse_args()


    '''      
    Data Assumptions:

    all data files should be tsv format with:
    1st column: index
    2nd column: sentence1, 
    3rd column: sentence2,
    last column: label
    '''

    pre_model_path = args.pre_model_path

    train_folder = args.train_folder
    dev_folder = args.dev_folder
    test_folder = args.test_folder
    train_data_filename = args.train_file
    dev_data_filename = args.dev_file  
    test_data_filename = args.test_file 
    hypothesis_only = str2bool(args.hypothesis_only)

    model_name = pre_model_path + "_" + \
                  train_data_filename + "_" + \
                  dev_data_filename + "_" + \
                  "lr_"+ str(args.learning_rate) + "_" + \
                  "wt_" + str(args.weight_decay) + "_" + \
                  "ws_" + str(args.warmup_steps) + "_" + \
                  str(hypothesis_only)

    model_output_path = "models/" + model_name

    results_file = "results/" + model_name + ".txt"

    ## It is assumed labels are named "not-entailed", and "entailed" in the data file
    num_labels = args.num_labels  
    train_batch_size = args.train_batch_size


    model_args = ModelArguments(
        model_name_or_path=pre_model_path,
    )

    training_args = TrainingArguments(
        output_dir=model_output_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=False,
        per_gpu_train_batch_size=train_batch_size,
        per_gpu_eval_batch_size=128,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        learning_rate = args.learning_rate,
        weight_decay = args.weight_decay,
	warmup_steps = args.warmup_steps,
        evaluate_during_training=True,
    )

    set_seed(training_args.seed)


    train_args = DataTrainingArguments(task_name="dnc2", data_dir=train_folder)
    dev_args = DataTrainingArguments(task_name="dnc2", data_dir=dev_folder)
    test_args = DataTrainingArguments(task_name="dnc2", data_dir=test_folder)

    # #### Configs
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="dnc2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )


    # Extract features from dataset
    train_dataset = GlueDataset(train_args, data_filename=train_data_filename,
                               num_labels = num_labels,
                               hypothesis_only = hypothesis_only,
                               tokenizer=tokenizer)
    dev_dataset = GlueDataset(dev_args, data_filename=dev_data_filename,
                               num_labels = num_labels,
                               hypothesis_only = hypothesis_only,
                               tokenizer=tokenizer)
    test_dataset = GlueDataset(test_args, data_filename=test_data_filename,
                           num_labels = num_labels,
                           hypothesis_only = hypothesis_only,
                           tokenizer=tokenizer)


    #### Initialize Trainer 



    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
            trainer.train(
                model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
            )
            trainer.save_model()
            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            if trainer.is_world_master():
                tokenizer.save_pretrained(training_args.output_dir)


    ### Predict
    #dev_p = trainer.predict(dev_dataset)
    #test_p = trainer.predict(test_dataset)

    #with open(results_file, 'w') as f1:
        #f1.write(f"Dev Accuracy: {compute_metrics(dev_p)}\n")
        #f1.write(f"Test Accuracy: {compute_metrics(test_p)}\n")


    ## Save dev and test predictions
    #save_obj(dev_p, model_output_path + "/dev_predictions.pkl")
    #save_obj(test_p, model_output_path + "/test_predictions.pkl")


if __name__ == "__main__":
    main()
