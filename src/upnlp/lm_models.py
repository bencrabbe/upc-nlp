"""
This file contains vanilla implementations of NNLM, RNNLM and Transformer-based language models
"""
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.functional import pad, dropout
import tqdm
import json, yaml
import nlp_datasets



__all__ = ['AutoModel','MarkovianLM','LstmLM','TransformerLM']



class AutoModel:
    """
    This is a namespace for easy loading of models defined from this module.
    """

    @staticmethod
    def from_pretrained(path,device='cpu'):
        """
        Loads a pretrained model from directory.

        Args:
            path(path) : path to the model directory (either in the hub or local)

        Returns:
            the pretrained model instance
        """
        hub = nlp_datasets.HubUpc()
        local_path = hub.get_local_path(path)
        yaml_path  = os.path.join(local_path,'config.yaml')
        with open(yaml_path) as infile:
            config = yaml.safe_load(infile)
        if config['family'] == 'markovian':
            return MarkovianLM.from_pretrained(local_path,device)
        elif config['family'] == 'recurrent':
            return LstmLM.from_pretrained(local_path,device)
        elif config['family'] == 'transformer':
            return TransformerLM.from_pretrained(local_path,device)
        else:
            raise Exception('Cannot load model: family type is unknown. ')


class PytorchLanguageModel(nn.Module):
    """
    This is an abstract class providing the generic interface to language models: train, eval and generate
    Actual model architectures are specified as subclasses.
    """
    def __init__(self,pad_value):
        super(PytorchLanguageModel,self).__init__()
        self.pad_value = pad_value

    @staticmethod
    def from_pretrained(path):
        """
        Loads a model from directory

        Args:
            path (path): a path to the model dir file

        Returns:
            a language model.
        """
        raise NotImplementedError


    def save_pretrained(self, path):
        """
        Saves the model parameters and hyperparameters to directory

        Args:
            path (path): path to the model directory
        """
        raise NotImplementedError

    def _reallocate_params(self,device):
        #supposes that all params are located into dedicated submodules
        for name,submodule in self.named_children():
            setattr(self,name,submodule.to(device))

    def get_device(self):
        """
        Returns the device on which all the parameters of the model are located

        Returns:
            device
        """
        device = next(self.parameters()).device
        if device == -1 :
            return 'cpu'
        return device

    def train_lm(self,train_loader,valid_loader,epochs,outdir=None,LR=0.001,device='cpu',show_progressbar=False):
        """
        Trains a model from scratch.

        Args:
            train_loader (dataloader): a dataloader on the train set

            valid_loader (dataloader): a dataloader on the valid set

            epochs (int): the number of training epochs

            outdir (path or string): path to the final model directory

        KwArgs:
            outdir (path or string): path to the final model directory. If set to None, no model is automatically saved to disk

            LR (float): the initial learning rate of AdamW

            device (str): the device on which to train the model ('cpu','cuda','mps' ...)

            show_progressbar (bool): whether to show the progressbar when training

        Returns
            the model with minimal loss over the epochs
        """
        self._reallocate_params(device)
        loss_fnc  = nn.CrossEntropyLoss(ignore_index = self.pad_value)
        optimizer = optim.AdamW(self.parameters(),lr=LR)

        min_loss = 100000000000000
        for e in range(epochs):
            loss_lst = []
            self.train()
            for batch in tqdm.tqdm(train_loader,disable= not show_progressbar):
                optimizer.zero_grad()
                X     = batch.to(device)
                Y     = X[:,1:]

                Yhat  = self.forward(X[:,:-1])
                N,S,_ = Yhat.shape

                loss  = loss_fnc(Yhat.view(N*S,-1), Y.reshape(-1))
                loss.backward()
                optimizer.step()
                loss_lst.append(loss.item())

            valid_loss = self.validate_lm(valid_loader,show_progressbar=show_progressbar)
            print(f'Epoch {e+1} | train loss = {sum(loss_lst)/len(loss_lst):.5f} | valid loss = {valid_loss:.5f}')

            if valid_loss <= min_loss and outdir is not None:
                self.save_pretrained(outdir)
                min_loss = valid_loss

        if outdir is not None:
            print(f'\ntraining done.\nModel saved in {outdir}')
            return self.from_pretrained(outdir,device=self.get_device())
        else:
            print(f'\ntraining done.')
            return self

    def validate_lm(self,valid_loader,perplexity=False,show_progressbar=False):
        """
        Computes the loss on the validation dataset

        Args:
            valid_loader (dataloader) : dataloader for the validation set

        KwArgs:
            perplexity (float): if true returns the perplexity of the language model on the validation set, otherwise an averaged loglikelihood

            show_progressbar (bool): whether to show the progressbar when training

        Returns :
            float. the averaged validation loss
        """
        device = self.get_device()
        loss_fnc  = nn.CrossEntropyLoss(ignore_index = self.pad_value)

        self.eval()
        loss_lst = []
        for batch in tqdm.tqdm(valid_loader,disable= not show_progressbar):
            X = batch.to(device)
            Y = X[:, 1:]

            Yhat = self.forward(X[:, :-1])
            N, S, _ = Yhat.shape

            if perplexity:
                mask       = (Y != self.pad_value)
                Yhat       = nn.LogSoftmax(dim=-1)(Yhat)
                preds_logp = Yhat.gather(-1,Y.unsqueeze(-1)).squeeze() * mask
                loss_lst.append( (torch.exp(-preds_logp.sum() / mask.sum())).item())
            else:
                loss = loss_fnc(Yhat.view(N * S, -1), Y.reshape(-1))
                loss_lst.append(loss.item())
        return sum(loss_lst)/len(loss_lst)


    def _predict(self,prefix,temperature,do_sample):
        """
        Generates the next token given the prefix.
        Args:
            prefix       (list): a list of token ids, the prefix symbol sequence
        KwArgs:
            temperature (float): the temperature of the softmax
            do_sample    (bool): whether to use sampling or argmax to generate next token
        Returns:
            int. the next token id
        """
        softmax = nn.Softmax(dim=0)
        X    = torch.LongTensor(prefix).unsqueeze(0).to(self.get_device())
        Yhat = self.forward(X).squeeze()[-1] / temperature
        if do_sample:
            return torch.multinomial(softmax(Yhat),1,False)
        else:
            return torch.argmax(Yhat)



    def generate(self,prefix,temperature=1.0,do_sample=True,max_tokens=40,eos_token_id= -1):
        """
        Generates symbols with the language model. This is a generic and inefficient method suitable for generating small texts with small models.
        Most larger scale models subclasses will want to override it.

        Args:
            prefix       (list): a list of token ids, the prefix symbol sequence

        KwArgs:
            temperature (float): the temperature of the softmax
            do_sample    (bool): whether to use sampling or argmax to generate next token
            max_tokens   (int) : the max number of generated tokens
            eos_token_id (int) : generation stops once this token_id is generated

        Returns:
            LongTensor. Tensor with the full encoded text
        """
        self.eval()
        if prefix[-1] == eos_token_id:
            prefix.pop()
        for _ in range(max_tokens):
            next_idx = int(self._predict(prefix,temperature,do_sample))
            prefix.append(next_idx)
            if next_idx == eos_token_id:
                return prefix
        return prefix

    def logprobs(self,sequence,bos_id):
        """
        Computes the logprob given context for each token given in the sequence.
        The method prepends a <bos> token at the beginning of the sequence if there is none
        Args:
            sequence (list) : a list of integer codes
            bos_id    (int) : code for the bos token
        Returns:
            tensor. a tensor of log probabilities of the same size as the input tensor.
            One value for each input token. No value for the bos token
        """
        if sequence[0] != bos_id:
            sequence = [bos_id] + sequence

        xinput = torch.LongTensor(sequence[:-1]).to(self.get_device())
        Y      = torch.LongTensor(sequence[1:]).to(self.get_device())
        Yhat = self.forward(xinput.unsqueeze(0))
        Yhat = nn.LogSoftmax(dim=-1)(Yhat)
        return Yhat.gather(-1, Y.unsqueeze(-1)).squeeze()

class LstmLM(PytorchLanguageModel):

    def __init__(self,vocab_size,emb_size,hidden_size,nlayers = 2,pad_value=0,dropout=0.0,weight_tying=True):
        """
        An LSTM language model.
        Args:
            vocab_size (int) : size of the vocabulary

            emb_size   (int) : size of the embeddings

            hidden_size (int): size of the hidden layer
        KwArgs:
            nlayers     (int): number of layers

            pad_value   (int): pad token id

            dropout   (float): value of the dropout
        """
        super(LstmLM,self).__init__(pad_value)
        self.E = nn.Embedding(vocab_size, emb_size, padding_idx=pad_value)
        self.lstm = nn.LSTM(emb_size, hidden_size, nlayers, batch_first=True, dropout=dropout)
        self.W = nn.Linear(hidden_size, vocab_size,bias=False)
        if weight_tying:
            self.W.weight = self.E.weight
        self.dropout = nn.Dropout(dropout)
        self.hyper = {'vocab_size':vocab_size,
                      'emb_size':emb_size,
                      'hidden_size':hidden_size,
                      'nlayers':nlayers,
                      'pad_value':pad_value,
                      'dropout':dropout}


    @staticmethod
    def from_pretrained(path,device='cpu'):
        """
        Loads a model from directory
        Args:
            path (path): a path to the model dir file
        KwArgs:
            device (device): device where to load the model
        Returns:
            a language model.
        """
        with open(os.path.join(path,'hyperparams.json')) as hyper:
            H = json.loads(hyper.read())
            model = LstmLM(**H).to(device)
            model.load_state_dict(torch.load(os.path.join(path,'params.pt'), weights_only=True))
        return model

    def save_pretrained(self,path):
        """
        Saves the model parameters and hyperparameters to directory
        Args:
            path (path): path to the model directory
        """
        torch.save(self.state_dict(), os.path.join(path,'params.pt'))
        with open(os.path.join(path,'hyperparams.json'),'w') as outfile:
            outfile.write(json.dumps(self.hyper))



    def forward(self,xinputs):
        """
        This takes a batch of N sequences of S tokens each.
        For each token it redicts its successors' scores

        Args:
            xinputs (tensor): the tokens ids, a tensor of shape (N,S)

        Returns:
            the scores. A tensor of shape (N,S,V)
        """
        embeddings = self.dropout(self.E(xinputs))
        hidden,_   = self.lstm(embeddings)
        return self.W(hidden)


class TransformerLM(PytorchLanguageModel):

    def __init__(self,vocab_size,hidden_size,nlayers = 2,nheads=4,pad_value=0,max_window_size=512,dropout=0.0,weight_tying=False):
        super(TransformerLM,self).__init__(pad_value)
        self.max_window_size = max_window_size
        self.word_embedding  = nn.Embedding(vocab_size,hidden_size,padding_idx=pad_value)
        self.posn_embedding  = nn.Embedding(max_window_size,hidden_size,padding_idx=pad_value)
        transformer_layer    = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nheads,batch_first=True,dropout=dropout)
        self.transformer     = nn.TransformerEncoder(transformer_layer, num_layers=nlayers)
        self.W               = nn.Linear(hidden_size,vocab_size)
        if weight_tying:
            self.W.weight = self.word_embedding.weight
        self.dropout         = nn.Dropout(dropout)
        self.hyper           = {'vocab_size':vocab_size,
                                'hidden_size':hidden_size,
                                'nlayers':nlayers,
                                'nheads':nheads,
                                'pad_value':pad_value,
                                'max_window_size':max_window_size,
                                'dropout':dropout}


    @staticmethod
    def from_pretrained(path, device='cpu'):
        """
        Loads a model from directory
        Args:
            path (path): a path to the model dir file
        KwArgs:
            device (device): the device where to load the weights
        Returns:
            a language model.
        """
        with open(os.path.join(path, 'hyperparams.json')) as hyper:
            H = json.loads(hyper.read())
            model = TransformerLM(**H).to(device)
            model.load_state_dict(torch.load(os.path.join(path, 'params.pt'), weights_only=True))
        return model

    def save_pretrained(self, path):
        """
        Saves the model parameters and hyperparameters to directory
        Args:
            path (path): path to the model directory
        """
        torch.save(self.state_dict(), os.path.join(path, 'params.pt'))
        with open(os.path.join(path, 'hyperparams.json'), 'w') as outfile:
            outfile.write(json.dumps(self.hyper))


    def forward(self,xinputs):
        N,S = xinputs.shape
        if S >= self.max_window_size:
            raise Exception(f"Problem when running TransformerLM. Maximum inference window size exceeded. Found {S} where maximum window size is {self.max_window_size}")

        dev = self.get_device()
        posn_ids = torch.arange(S,device=dev).expand(N,S)
        input_embeddings = self.dropout(self.word_embedding(xinputs) + self.posn_embedding(posn_ids))

        causal_mask  = nn.Transformer.generate_square_subsequent_mask(S,dev)
        padding_mask = (xinputs == self.hyper['pad_value']).to(dev)

        hidden = self.transformer(input_embeddings,mask=causal_mask,is_causal=True,src_key_padding_mask=padding_mask)
        return self.W(hidden)


class MarkovianLM(PytorchLanguageModel):

    def __init__(self,ctx_size,vocab_size,emb_size,hidden_size,pad_value=0,dropout=0.):

        super(MarkovianLM,self).__init__(pad_value)
        self.ctx_size  = ctx_size
        self.E   = nn.Embedding(vocab_size,emb_size,padding_idx=pad_value)
        self.Win = nn.Linear(ctx_size*emb_size,hidden_size)
        self.act = nn.Tanh()
        self.Wout = nn.Linear(hidden_size, vocab_size,bias=False)
        self.dropout = nn.Dropout(dropout)

        self.hyper = {  'ctx_size' : ctx_size,
                        'vocab_size': vocab_size,
                        'emb_size': emb_size,
                        'hidden_size': hidden_size,
                        'pad_value': pad_value,
                        'dropout':dropout  }


    @staticmethod
    def from_pretrained(path, device='cpu'):
        """
        Loads a model from directory
        Args:
            path (path): a path to the model dir file
        KwArgs:
            device (device): the device where to load the weights
        Returns:
            a language model.
        """
        with open(os.path.join(path, 'hyperparams.json')) as hyper:
            H = json.loads(hyper.read())
            model = MarkovianLM(**H).to(device)
            model.load_state_dict(torch.load(os.path.join(path, 'params.pt'), weights_only=True))
        return model

    def save_pretrained(self, path):
        """
        Saves the model parameters and hyperparameters to directory
        Args:
            path (path): path to the model directory
        """
        torch.save(self.state_dict(), os.path.join(path, 'params.pt'))
        with open(os.path.join(path, 'hyperparams.json'), 'w') as outfile:
            outfile.write(json.dumps(self.hyper))

    def forward(self,xinputs):
        """
        This takes a batch of N sequences of S tokens each
        And for each token predicts its successors' scores
        Args:
           xinputs (tensor): the tokens ids, a tensor of shape (N,S)
        Returns:
           the scores. A tensor of shape (N,S,V)
        """
        xinputs    = pad(xinputs,(self.ctx_size-1,0), value = self.pad_value)
        xinputs    = xinputs.unfold(dimension=-1,size=self.ctx_size,step=1)
        embeddings = self.dropout(self.E(xinputs))
        return self.Wout(self.act(self.dropout(self.Win(torch.flatten(embeddings,start_dim=-2)))))



if __name__ == '__main__':

#Simple howto

    import tokenizers
    import nlp_datasets
    from torch.utils.data import DataLoader


    zebra = """
There are five houses.
The Englishman lives in the red house.
The Spaniard owns the dog.
Coffee is drunk in the green house.
The Ukrainian drinks tea.
The green house is immediately to the right of the ivory house.
The Old Gold smoker owns snails.
Kools are smoked in the yellow house.
Milk is drunk in the middle house.
The Norwegian lives in the first house.
The man who smokes Chesterfields lives in the house next to the man with the fox.
Kools are smoked in the house next to the house where the horse is kept.
The Lucky Strike smoker drinks orange juice.
The Japanese smokes Parliaments.
The Norwegian lives next to the blue house.
"""
    model_dir = "zebra"
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    zebra       = tokenizers.normalize_text(zebra)
    segmenter   = tokenizers.PunctuationTokenizer(zebra.split(),bos='<bos>',eos='<eos>')
    segmenter.save_pretrained(model_dir)
    dataset     = nlp_datasets.RandomAccessRawText([tokenizers.normalize_text(sent) for sent in zebra.split('\n') if sent and not sent.isspace()],segmenter)
    trainloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=segmenter.pad_batch)
    validloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=segmenter.pad_batch)

    model       = MarkovianLM(3,segmenter.vocab_size,64,32,pad_value=segmenter.pad_id)
    #model      = LstmLM(segmenter.vocab_size,64,64,pad_value=segmenter.pad_id)
    #model      = TransformerLM(segmenter.vocab_size, 128 , nlayers=2, nheads=4,pad_value=segmenter.pad_id, max_window_size=128, dropout=0.0)
    model       = model.train_lm(trainloader,validloader,150,model_dir,LR=0.001,device="mps")
    print('perplexity',model.validate_lm(validloader,perplexity=True))
    gen_idxes = model.generate(segmenter("The man who smokes Chesterfields"),eos_token_id=segmenter.eos_token_id)
    print(segmenter.decode(gen_idxes))


