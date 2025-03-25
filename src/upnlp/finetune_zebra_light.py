"""
This file provides examples on how to fine tune lightweight language models like GPT-2
"""

# Fine tune GPT-2 on zebra logic puzzles
# here we choose to mostly use torch to do so

import torch
import json
import tqdm
import torch.nn as nn

from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


class ZebraDataset(Dataset):

    def __init__(self,zebra_dataset,tokenizer,train_mode=True):

        self.xsentences = []
        self.train_mode = train_mode
        self.tokenizer  = tokenizer

        for example in zebra_dataset:
            if train_mode:
                subwords = tokenizer.apply_chat_template(example['messages'],chat_template=chatML_tpl,tokenize=True)
                self.xsentences.append(subwords)

            else: #test mode 
                
                prompt_string = tokenizer.apply_chat_template(example['messages'][:2], #we drop the assistant answer here
                                                         chat_template=chatML_tpl,
                                                         tokenize=False,
                                                         add_generation_prompt=False)
                answer = example['messages'][-1]['content']
                self.xsentences.append( (prompt_string,answer))

        self.pad_tok_id = tokenizer.eos_token_id


    def __len__(self):

        return len(self.xsentences)


    def __getitem__(self, idx):

        if self.train_mode:
            return self.xsentences[idx]
        else:
            prompt_str, answer = self.xsentences[idx]
            xinputs       = self.tokenizer(prompt_str,return_tensors="pt")
            return (xinputs,prompt_str,answer) 

    def pad_batch(self,itemlist):
        """
        Called by the dataloader to batch the data. Batches should not be truncated otherwise we risk to lose the
        connection between the premises and the answers. For the logic riddles batches are in practice < 1024
        """
        assert(self.train_mode)
        xbatch = [x for x in itemlist]
        maxseq = max([len(x) for x in xbatch])
        xbatch = torch.LongTensor( [ x + [self.pad_tok_id]*(maxseq-len(x)) for x in xbatch ] )
        return xbatch


#jinja template formatting dict elements into strings (used by the apply_template of the tokenizer)
chatML_tpl="""
{% if messages[0]['role'] == 'system' %}
    {% set offset = 1 %}
{% else %}
    {% set offset = 0 %}
{% endif %}
{{ bos_token }}
{% for message in messages %}
    {{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|im_end|>\n' }}
    {{ eos_token }}
{% endfor %}
{% if add_generation_prompt %}
    {{ '<|im_start|>assistant\n' }}
{% endif %}
"""

def zebra_fine_tune(causal_model,trainloader,pad_id,lr=0.0001,epochs=3,device='cuda'):

      #todo: add epochwise validation

      cross_entropy = nn.CrossEntropyLoss(ignore_index = pad_id)
      optimizer     = torch.optim.AdamW(causal_model.parameters(), lr=lr)


      for epoch in range(epochs):

          loss_lst = []

          for batch in tqdm.tqdm(trainloader,total=len(trainloader)):

              optimizer.zero_grad()

              xbatch = batch[:,:-1].to(device)
              ybatch = batch[:,1:].to(device)

              pred_logits = causal_model(xbatch).logits
              batch,seq  = ybatch.shape

              loss       = cross_entropy(pred_logits.reshape(batch*seq,-1),ybatch.reshape(batch*seq))

              #backward pass
              loss.backward()
              loss_lst.append(loss.item())

              #update and reset
              optimizer.step()

          print(f'Epoch {epoch}, loss {sum(loss_lst)/len(loss_lst)}')



def format_prediction(pred_str,start_tag='<|im_start|>assistant',end_tag='<|im_end|>'):
      """
      Extracts the solution from the model generated text
      """
      start = pred_str.find(start_tag)
      end   = pred_str.find(end_tag)

      if start >= 0 and end >=0 :
            pred_str = pred_str[start+len(start_tag):end]
      return ' '.join(pred_str.split())#normalizes whitespace


def eval_fnc(tokenizer,model,test_loader,device):

  #drops a bunch of annoying warnings
  from transformers.utils import logging
  logging.set_verbosity_error() 

  acc, N = 0,0
  for prompt,prompt_str,answer in test_loader:
        xinputs       = prompt.to(device)
        generated_ids = model.generate(**xinputs, max_new_tokens=100, do_sample=True)
        pred_answer   = tokenizer.decode(generated_ids[0])
        pred_answer   = format_prediction(pred_answer[len(prompt_str):])
        N   += 1
        acc += (pred_answer == answer) #could be relaxed
        print(pred_answer,'***',answer, 'v' if pred_answer == answer else 'x')
  return acc/N


def finetune_and_eval(model_path="zebra_params.pt",train_file=None,valid_file=None,test_file=None,device='cuda',epochs=3):

    """
    Args:
        modelfile: file where to save and or load the fine tuned model parameters
    """ 
    train_only   = test_file is None and valid_file and train_file
    train_n_test = test_file and valid_file and train_file
    test_only    = test_file is not None

    if train_only or train_n_test:
        print('Training...')
        tokenizer    = AutoTokenizer.from_pretrained("gpt2")
        model        = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).to(device)

        with open(train_file) as intrain:
            train = json.loads(intrain.read())
        with open(valid_file) as invalid:
            valid = json.loads(invalid.read())

        train_set     = ZebraDataset(train,tokenizer)
        train_loader  = DataLoader(train_set, batch_size=2, shuffle=True,collate_fn=train_set.pad_batch)

        #valid_set     = ZebraDataset(valid,tokenizer,train_mode=False)
        #valid_loader   = DataLoader(valid_set, batch_size=2, shuffle=True,collate_fn=valid_set.pad_batch)
        #not used for now

        zebra_fine_tune(model,train_loader,tokenizer.eos_token_id,device=device,epochs=epochs)
        model.save_pretrained(model_path)


    if train_n_test or test_only:
        print('Testing...')

        tokenizer    = AutoTokenizer.from_pretrained("gpt2")
        model        = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32).to(device)

        with open(test_file) as intest:
            test     = json.loads(intest.read())
            print(len(test))
        test_set     = ZebraDataset(test,tokenizer,train_mode=False)
        test_loader  = DataLoader(test_set, batch_size=1, shuffle=False,collate_fn=lambda x:x[0])
        acc = eval_fnc(tokenizer,model,test_loader,device)
        print(f'Model accurracy on test {acc}')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                    prog='Reasoning dataset generator',
                    description='Generates artificial data sets')

    parser.add_argument('model_path')   
    parser.add_argument('--train_path',default=None)
    parser.add_argument('--valid_path',default=None)
    parser.add_argument('--test_path',default=None)
    parser.add_argument('--train_epochs',type=int,default=3)
    args = parser.parse_args()

    finetune_and_eval(model_path=args.model_path,
                      train_file=args.train_path,
                      valid_file=args.valid_path,
                      test_file=args.test_path,
                      epochs=args.train_epochs,
                      device='cuda')

