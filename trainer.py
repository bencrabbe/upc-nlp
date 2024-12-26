from importlib.metadata import pass_none

import torch
import yaml
import os
import optuna
import lm_models
import tokenizers
import nlp_datasets
import argparse
from collections import Counter



def transformer(trial,model_dir,segmenter,trainloader,validloader,ctx_size,epochs,device):
    """
    Optuna setup for training and evaluating a recurrent model.
    Currently it cannot be run in parallel.
    Args:
        trial (trial)            : an optuna trial
        model_dir (path)         : path where to store the trained model
        segmenter (Tokenizer)    : a tokenizer
        model (LSTM Model)  : a markovian model
        trainloader (DataLoader) : a pytorch dataloader
        validloader (DataLoader) : a pytorch dataloader
        epochs (int)             : the budget for training epochs
        device      (device)     : a device where to run the trial
    Returns:
        perplexity score
    """
    hidden_size = trial.suggest_categorical('hidden_size', (256, 512, 768))
    nheads      = trial.suggest_categorical('nheads', (2,4,8))
    nlayers     = trial.suggest_categorical('nlayers', (2,4,6))
    lr          = trial.suggest_float('LR', 0.00001, 0.01, log=True)
    dropout     = trial.suggest_float('dropout', 0.1, 0.5)

    model = lm_models.TransformerLM(segmenter.vocab_size,hidden_size,nlayers = nlayers,nheads=nheads,pad_value=segmenter.pad_id,max_window_size=512,dropout=dropout)
    model = model.train_lm(trainloader, validloader, epochs, model_dir, LR=lr, device=device)
    return model.validate_lm(validloader, perplexity=True)



def recurrent(trial,model_dir,segmenter,trainloader,validloader,ctx_size,epochs,device):
    """
    Optuna setup for training and evaluating a recurrent model.
    Currently it cannot be run in parallel.
    Args:
        trial (trial)            : an optuna trial
        model_dir (path)         : path where to store the trained model
        segmenter (Tokenizer)    : a tokenizer
        model (LSTM Model)  : a markovian model
        trainloader (DataLoader) : a pytorch dataloader
        validloader (DataLoader) : a pytorch dataloader
        epochs (int)             : the budget for training epochs
        device      (device)     : a device where to run the trial
    Returns:
        perplexity score
    """
    emb_size = trial.suggest_categorical('emb_size', (128, 256, 512))
    lr = trial.suggest_float('LR', 0.0001, 0.1, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    nlayers = trial.suggest_categorical('nlayers', (1, 2, 4))
    model = lm_models.LstmLM(segmenter.vocab_size, emb_size, emb_size, nlayers=nlayers,pad_value=segmenter.pad_id, dropout=dropout)
    model = model.train_lm(trainloader, validloader, epochs, model_dir, LR=lr, device=device)
    return model.validate_lm(validloader, perplexity=True)


def markovian(trial,model_dir,segmenter,trainloader,validloader,ctx_size,epochs,device):
    """
    Optuna setup for training and evaluating a model.
    Currently it cannot be run in parallel.
    Args:
        trial (trial)            : an optuna trial
        model_dir (path)         : path where to store the trained model
        segmenter (Tokenizer)    : a tokenizer
        model (Markovian Model)  : a markovian model
        trainloader (DataLoader) : a pytorch dataloader
        validloader (DataLoader) : a pytorch dataloader
        ctx_size    (int)        : size of the context for this markovian model
        epochs (int)             : the budget for training epochs
        device      (device)     : a device where to run the trial
    Returns:
        perplexity score
    """
    emb_size    = trial.suggest_categorical('emb_size',(128,256,512))
    lr          = trial.suggest_float('LR',0.0001,0.1,log=True)
    dropout     = trial.suggest_float('dropout',0.1,0.5)

    model       = lm_models.MarkovianLM(ctx_size,segmenter.vocab_size,emb_size,round(emb_size/2),pad_value=segmenter.pad_id,dropout=dropout)
    model       = model.train_lm(trainloader,validloader,epochs,model_dir,LR=lr,device=device)
    return model.validate_lm(validloader,perplexity=True)



if __name__ == '__main__':

    from torch.utils.data import DataLoader


    parser = argparse.ArgumentParser(prog="""Language model training script. By default the script searches for decent hyperparameter values.
    One can train a model with predefined hyperparameters values using a yaml parameter file given as config argument""")
    parser.add_argument('trainfile',help='path to training file')
    parser.add_argument('validfile',help='path to validation file')
    parser.add_argument('-c','--config',help='path to the yaml config file')
    parser.add_argument('-m','--model_name',help='path to the model dir')
    parser.add_argument('-d','--device',default='cpu',help='device identifier')
    parser.add_argument('-e','--epochs',default=20,type=int,help='num epochs for each trial')
    parser.add_argument('-f','--family',default='markovian',help='model family in {markovian,recurrent,transformer}')
    parser.add_argument('-b','--batch_size',default=8,type=int,help='batch size')
    parser.add_argument('--ctx_size',default=2,type=int,help='context size (for markovian models only)')
    parser.add_argument('--max_vocab_size',default=50000,type=int,help='maximum vocabulary size')

    args = parser.parse_args()

    if not os.path.isdir(args.model_name):
        os.mkdir(args.model_name)

    with open(args.trainfile) as train:
        traintext = tokenizers.normalize_text(train.read())
    with open(args.validfile) as valid:
        validtext = tokenizers.normalize_text(valid.read())

    counter     = Counter(traintext.split())
    print('True vocabulary size:',len(counter), 'Max vocab size:',args.max_vocab_size)
    segmenter   = tokenizers.DefaultTokenizer([elt for elt, count in counter.most_common(args.max_vocab_size)],bos='<bos>',eos='<eos>')

    trainset = nlp_datasets.RandomAccessRawText([sent for sent in traintext.split('\n') if sent and not sent.isspace()],segmenter)
    validset = nlp_datasets.RandomAccessRawText([sent for sent in validtext.split('\n') if sent and not sent.isspace()],segmenter)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=segmenter.pad_batch)
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=True, collate_fn=segmenter.pad_batch)

    study = optuna.create_study()
    if args.family == 'markovian':
        study.optimize(lambda x:markovian(x,args.model_name,segmenter, trainloader, validloader,args.ctx_size,args.epochs,args.device),n_trials=20,n_jobs=1)
        print("*** Search summary ***")
        print('Best parameters for markovian model with context size', args.ctx_size)
        print('\n'.join([f'{key} : {value}' for key, value in study.best_params.items()]))
        print()
    elif args.family == 'recurrent':
        study.optimize(lambda x: recurrent(x, args.model_name, segmenter, trainloader, validloader, args.ctx_size, args.epochs,args.device), n_trials=20, n_jobs=1)
        print("*** Search summary ***")
        print('Best parameters for recurrent (LSTM) model')
        print('\n'.join([f'{key} : {value}' for key, value in study.best_params.items()]))
        print()
    elif args.family == 'transformer':
        study.optimize(lambda x: transformer(x, args.model_name, segmenter, trainloader, validloader, args.ctx_size, args.epochs,args.device), n_trials=20, n_jobs=1)
        print("*** Search summary ***")
        print('Best parameters for transformer model')
        print('\n'.join([f'{key} : {value}' for key, value in study.best_params.items()]))
        print()
    else:
        print('Model family',args.family, 'is unknown. Aborting')

