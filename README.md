# integrity_verification

This is the code for the article *Integrity Verification for Scientific Papers: The first exploration of the text*.

## Dataset

* INTER-WCL
* INTER-AI
* INTER-SYM
  
## Base Model

* bert-base-uncased   (https://huggingface.co/bert-base-uncased) 
* scibert-base-uncased   (https://huggingface.co/allenai/scibert_scivocab_uncased)

## Train 

You can start training by setting the **base model** and **dataset** 
```
python train.py --bert_model model_name --dataset_mode dataset_name
```
