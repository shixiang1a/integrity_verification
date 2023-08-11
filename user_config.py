import argparse
import os
import torch


def config_init():
     parser = argparse.ArgumentParser()

     # required arguments

     parser.add_argument("--bert_model", default='allenai/scibert_scivocab_uncased', type=str,
     help="Bert pre-trained model selected in the list: bert-base-uncased, "
     "bert-large-uncased, bert-base-chinese, hfl/chinese-roberta-wwm-ext, hfl/chinese-macbert-large, allenai/scibert_scivocab_uncased")
     
     parser.add_argument("--batch_size", default=4, type=int)
     parser.add_argument("--eval_batch_size", default=1, type=int)
     parser.add_argument("--hidden_dropout", default=0.1, type=float)
     parser.add_argument("--learning_rate", default=5e-5, type=float)
     parser.add_argument("--core_learning_rate", default=5e-5, type=float)
     parser.add_argument("--epoch", default=30, type=int)
     parser.add_argument("--hidden_size", default=768, type=int)
     parser.add_argument("--windows", default=0, type=int)
     parser.add_argument("--max_grad_norm", default=1.0, type=float)
     parser.add_argument("--width_emb_size", default=25, type=int)
     parser.add_argument("--lr_warmup", default=0.1, type=int)
     parser.add_argument("--weight_decay", default=0.01, type=float)
     parser.add_argument("--max_span_len", default=40, type=int)
     parser.add_argument("--device", default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), type=torch.device, required=False)

     # require user to define
     parser.add_argument("--log_path", default='./log/', type=str)
     parser.add_argument("--model_path", default='./model/', type=str)
     parser.add_argument("--checkpoint_path", default='./checkpoint/', type=str)

     # special model config
     # biaffine
     parser.add_argument("--biaffine_output_features", default=128, type=int)

     # data config
     parser.add_argument("--data_mode", default='INTER_SYM', choices=['INTER_WCL', 'INTER_SYM', 'INTER_AI'], type=str)
     parser.add_argument("--dataset", default='./dataset/', type=str)

     # train config
     parser.add_argument("--train_mode", default='I', type=str, choices=['I', 'C'], help='continue training: C/ start a new training: I')

     # model choose
     parser.add_argument("--model_mode", default='InterF', type=str, choices=['InterF', 'InterF_ent'], help='Integrity/NER model')

     # model label
     parser.add_argument("--num_labels", default=6, type=int)
     parser.add_argument("--sent_labels", default=4, type=int)

     # predict
     parser.add_argument('--predict_mode', default='T', type=str, choices=['P', 'T'], help='test and predict the results: P-only predict; T-only test if we have test set')
     parser.add_argument('--predict_path', default='./dataset/INTER/test_split.json', type=str)
     parser.add_argument('--result_path', default='./result/', type=str)

     # test mode
     parser.add_argument('--test_mode', default='test', type=str)

     # early_stopping
     parser.add_argument('--use_es', default=False, type=bool)

     # language
     parser.add_argument('--lan', default='en', choices=['en', 'cn'], type=str)

     # context
     parser.add_argument('--context', default='o', choices=['r', 'o', 's'], type=str)

     # sent_cut
     parser.add_argument('--sent_mode', default='cut', choices=['cut', 'split'], type=str)

     # split sent
     parser.add_argument('--split_sent', default=False, type=bool)

     # negative span sample
     parser.add_argument('--sample_count', default=500, type=int)

     # data to transform
     parser.add_argument('--trans_dataset', default='GENIA', type=str)

     # seed 42 / 665917845
     parser.add_argument('--random_seed', default=4, type=int)

     # save mode span / sent
     parser.add_argument('--save_mode', default='sent', type=str)

     
     args = parser.parse_args()

     return args