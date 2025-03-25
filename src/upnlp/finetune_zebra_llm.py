import os
import json
import torch
import pprint

#hf stuff
import transformers,trl
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset

from trl import SFTConfig,SFTTrainer
from peft import LoraConfig, PeftModel, get_peft_model,prepare_model_for_kbit_training


"""
This illustrates how to fine tune a model of the gemma series (Gemma 1&2 et it)
@see https://huggingface.co/google/gemma-2b-it
"""

#@see for dataset formatting https://medium.com/the-ai-forum/instruction-fine-tuning-gemma-2b-on-medical-reasoning-and-convert-the-finetuned-model-into-gguf-844191f8d329

def apply_gemma_template(example,train_mode=True,start='<start_of_turn>',end='<end_of_turn>'):
    """
    Maps a data dict to a gemma formatted prompt
    """
    bos = 'We want a very short answer to the following riddle'
    formatted = [bos]
    for msg in example['messages']:
        role = 'user' if msg['role'] == 'user' else 'model'
        if train_mode:
            formatted.append(f"{start}{role}\n{msg['content']}\n{end}")
        else:
            if role == 'model':
               formatted.append(f'{start}{role}\n')
            else:      
               formatted.append(f"{start}{role}\n{msg['content']}\n{end}")

    return {'prompt':'\n'.join(formatted)}


def format_gemma_prediction(pred_str,start_tag='<start_of_turn>model',end_tag='<end_of_turn>'):

    start = pred_str.find(start_tag)
    end   = pred_str.find(end_tag,start+1)

    if start >= 0 and end >=0 :
        pred_str = pred_str[start+len(start_tag):end]
    return ' '.join(pred_str.split())#normalizes whitespace

def eval_fnc(tokenizer,model,testset,device):

  #drops a bunch of annoying warnings
  from transformers.utils import logging
  logging.set_verbosity_error() 

  acc, N = 0,0
  for prompt in testset:
        answer        = prompt['messages'][1]['content']
        xinputs       = {'input_ids' : torch.tensor(prompt['input_ids'],device=device).unsqueeze(0),
                         'attention_mask' : torch.tensor(prompt['attention_mask'],device=device).unsqueeze(0)}
        generated_ids = model.generate(**xinputs, max_new_tokens=100, do_sample=True)
        pred_answer   = tokenizer.decode(generated_ids[0])
        pred_answer   = format_gemma_prediction(pred_answer)
        N   += 1
        acc += (pred_answer == answer) #could be relaxed
        print(pred_answer,'***',answer, 'v' if pred_answer == answer else 'x')
  return acc/N


def finetune_and_eval(model_path="params_gemma",train_file=None,valid_file=None,test_file=None,device='cuda',max_steps=1200,max_seq_length=1024,batch_size=4):

        base_model_id = "google/gemma-2-2b-it"


        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.bfloat16,
                                             bnb_4bit_quant_type="nf4")

        tokenizer        = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model            = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                            torch_dtype=torch.bfloat16,
                                                            #device_map={"": 0},
                                                            quantization_config=quantization_config, 
                                                            attn_implementation='eager').to(device)
    
        if train_file:
            trainset = Dataset.from_json(train_file)

            print(tokenizer.apply_chat_template(trainset[0]['messages'],tokenize=False))

            #exit(0)

            print("Example training item")
            pprint.pprint(trainset[0])
            print(tokenizer.apply_chat_template(trainset[0]['messages'],tokenize=False))

            #maps data to tensors
            #trainset = trainset.map(apply_gemma_template, fn_kwargs={'train_mode':True})
            #trainset = trainset.map(lambda x : tokenizer(x['prompt']),batched=True).remove_columns('messages')

            print(trainset)

            lora_config  = LoraConfig( r=64, #the rank (also impacts memory consumption)
                                       target_modules="all-linear",
                                       lora_alpha=64, #the more weight the more the base model is updated
                                       #target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                                       bias="all",
                                       task_type="CAUSAL_LM")
        
            #model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)

            training_args = SFTConfig(per_device_train_batch_size=batch_size,
                                  gradient_accumulation_steps=1,
                                  warmup_steps=100,
                                  max_steps=max_steps,
                                  learning_rate=1e-4,
                                  bf16=True,
                                  logging_steps=10,
                                  max_seq_length=max_seq_length,
                                  #output_dir="model_path",
                                  optim="paged_adamw_8bit")

    
    
            trainer   = SFTTrainer(model=model,
                               train_dataset=trainset,
                               #eval_dataset=testset,
                               #dataset_text_field="messages",
                               args=training_args,
                               peft_config=lora_config,
                               data_collator=trl.DataCollatorForCompletionOnlyLM(response_template='model', tokenizer=tokenizer))
                               #data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
    
            print('\ntraining.')
            trainer.train()
        
            print('merging and saving.')
        
            ft_model = model_path+'_ft'
            trainer.model.save_pretrained(ft_model)


            #Merges Lora updates into main model
            base_model            = AutoModelForCausalLM.from_pretrained(base_model_id,
                                                                low_cpu_mem_usage=True,
                                                                torch_dtype=torch.float16,
                                                                return_dict=True,
                                                                device_map=device)
    
            merged_model = PeftModel.from_pretrained(base_model,ft_model)
            merged_model = merged_model.merge_and_unload()

            # Saves the merged model
            merged_model.save_pretrained(model_path+'_merged', safe_serialization=True)
            tokenizer.save_pretrained(model_path+'_merged')


            print(f'done.\nmerged model saved as {model_path}_merged')


        if test_file:

            ft_model = model_path+'_merged'

            testset  = Dataset.from_json(test_file)
            testset  = testset.map(apply_gemma_template, fn_kwargs={'train_mode':False})
            testset  = testset.map(lambda x : tokenizer(x['prompt']),batched=True)

            tokenizer = AutoTokenizer.from_pretrained(ft_model)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(ft_model,attn_implementation='eager').to(device)
            print(eval_fnc(tokenizer,model,testset,device))



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
                    prog='Reasoning dataset generator',
                    description='Generates artificial data sets')

    parser.add_argument('model_path')   
    parser.add_argument('--train_path',default=None)
    parser.add_argument('--valid_path',default=None)
    parser.add_argument('--test_path',default=None)
    parser.add_argument('--train_steps',type=int,default=1200)
    parser.add_argument('--gpu_id',type=int,default=0)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--max_seq_length',type=int,default=1024)


    args = parser.parse_args()


    hf_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu_id)

    finetune_and_eval(model_path="zebra_params_gemma.pt",
                      train_file=args.train_path,
                      valid_file=args.valid_path,
                      test_file=args.test_path,
                      device=f'cuda:{args.gpu_id}',
                      max_steps=args.train_steps,
                      max_seq_length=args.max_seq_length,
                      batch_size=args.batch_size)



