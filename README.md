# Integrity Verification (INTEGER)

This is the code for the article *Integrity Verification for Scientific Papers: The first exploration of the text*.

![markdown picture](https://github.com/shixiang1a/integrity_verification/blob/main/model_INTEGER.png)

## Dataset

* **INTEGER-WCL**  Multidisciplinary domain dataset. It contains only one type of term and its corresponding definition.
* **INTEGER-AI**   AI domain dataset. It contains one type of term (i.e., symbol) and the corresponding value and definition description.
* **INTEGER-SYM**  AI domain dataset. It contains multiple types of terms (e.g., method, application) and various descriptions (e.g., definition, characteristic, statement).
  
## Base Model

* bert-base-cased   (https://huggingface.co/bert-base-cased) 
* scibert-base-uncased   (https://huggingface.co/allenai/scibert_scivocab_uncased)

## Enviroment
```
python=3.8.12
pytorch=2.0.1
transformers=4.24.0
cudatoolkit=11.8
numpy=1.24.4
```

## Train 

You can start training by setting the base model and dataset 
```
python train.py --bert_model model_name --dataset_mode dataset_name
```
Also, you can change the model for **INTEGER** or **NER**
```
python train.py --model_mode InterF or InterF_ent
```

## Predict
Use the best model to predict the result
```
python predict.py
```
