import yaml
import lm_models
import argparse
import datasets


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Language model training script')
    parser.add_argument('trainfile',help='path to training file')
    parser.add_argument('validfile',help='path to validation file')
    parser.add_argument('-c','--config',help='path to the yaml config file')
    parser.add_argument('-m','--model_name',help='path to the model dir')

    #args = parser.parse_args()

    from itertools import product

    emb_size =  [256,512]
    lr       = [0.1,0.01,0.001]
    dropout  = [0.1,0.2,0.3]
    epochs   = [2]
    max_vocab_size = [200000]
    print(list(product(emb_size,lr,dropout,epochs,max_vocab_size)))

