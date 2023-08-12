# Integrity Verification (INTEGER)

This is the code for the article *Integrity Verification for Scientific Papers: The first exploration of the text*.

## Dataset

* INTEGER-WCL
* INTEGER-AI
* INTEGER-SYM
  
## Base Model

* bert-base-cased   (https://huggingface.co/bert-base-cased) 
* scibert-base-uncased   (https://huggingface.co/allenai/scibert_scivocab_uncased)

## Train 

You can start training by setting the base model and dataset 
```
python train.py --bert_model model_name --dataset_mode dataset_name
```
Also, you can change the model for **INTER** or **NER**
```
python train.py --model_mode InterF or InterF_ent
```

## Predict
Use the best model to predict the result
```
python predict.py
```
