# integrity_verification

## Dataset

* INTER-WCL
* INTER-AI
* INTER-SYM
  
## Base Model

* Bert-base-uncased   (https://huggingface.co/bert-base-uncased) 
* Scibert-base-uncased   (https://huggingface.co/allenai/scibert_scivocab_uncased)

## Train 

You can start training by setting the **base model** and **dataset** \
`python train.py --bert_model model_name --dataset_mode dataset_name`
