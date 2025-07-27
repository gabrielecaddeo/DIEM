"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_stsbenchmark.py

OR
python training_stsbenchmark.py pretrained_transformer_model_name
"""

import logging
import sys
import traceback
from datetime import datetime

from datasets import load_dataset

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from diem_functions import DIEM_Stat
import numpy as np

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) 
# You can specify any Hugging Face pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-mpnet-base-v2"
train_batch_size = 16
num_epochs = 10

output_dir = (
    "output/training_stsbenchmark_" + model_name.replace("/", "-") + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)

# 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
train_dataset = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(train_dataset)
print(train_dataset[:10])
# print(train_dataset['sentence2'][:10])
emb1 = model.encode(train_dataset['sentence1'], normalize_embeddings=False)
emb2 = model.encode(train_dataset['sentence2'], normalize_embeddings=False)

minV1 = np.min(emb1)
maxV1 = np.max(emb1)
minV2 = np.min(emb2)
maxV2 = np.max(emb2)
if minV1 < minV2:
    minV = minV1
else:
    minV = minV2
if maxV1 > maxV2:
    maxV = maxV1
else:
    maxV = maxV2
print(f"MinV: {minV}, MaxV: {maxV}")
# distances = np.linalg.norm(emb1 - emb2, axis=1)
# exp_center = np.median(distances)
# vard = np.var(distances)
# 3. Define our training loss
# CosineSimilarityLoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) needs two text columns and one
# similarity score column (between 0 and 1)
# exp_center, vard, std_one, orth_med, min_DIEM, max_DIEM = DIEM_Stat(384, maxV, minV, fig_flag=0)
# max_DIEM, min_DIEM, exp_center, vard = 199.23838672551682, -137.3673494631949, 29.400150417954006, 0.7868427873208158 # allminilm, -80
max_DIEM, min_DIEM, exp_center, vard = 277.53691709794305, -191.35667730511594, 15.147114194067642, 0.10601473846729577 # all-mpnet-base-v2, -130
import torch
DIEM = torch.diag((torch.cdist(torch.tensor(emb1), torch.tensor(emb2)) - exp_center) / vard) * (maxV - minV)
print(DIEM[:10])
print(f"Max DIEM: {max_DIEM}, Min DIEM: {min_DIEM}, exp_center: {exp_center}, vard: {vard}")
# print(exp_center, vard)
# exit(0)
train_loss = losses.DIEMLoss(model=model, exp_center=exp_center, vard=vard, maxV=maxV, minV=minV, min_labels=min_DIEM, max_labels=-130)
# train_loss = losses.CoSENTLoss(model=model)
model.exp_center = exp_center
model.vard = vard
model.minV = minV
model.maxV = maxV
model.min_labels = min_DIEM
model.max_labels = -130
# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.DIEM,
    similarity_fn_names=["diem"],
    name="sts-dev",
)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="sts",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
print('done')
trainer.train()
print('after train')

# 7. Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.DIEM,
    similarity_fn_names=["diem"],
    name="sts-test",
)
test_evaluator(model)

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"{model_name}-sts")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-sts')`."
    )