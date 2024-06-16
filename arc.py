import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import numpy as np
from tqdm import tqdm
# import argparse

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (
    LogitsProcessorList,
    MinNewTokensLengthLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    AutoTokenizer, 
    AutoModel,
    RobertaForSequenceClassification,
    GenerationConfig, LlamaTokenizer, LlamaForCausalLM,
    AutoModelForSequenceClassification
)

from transformers import logging
logging.set_verbosity_error()

batch_size = 1
dev_batch_size = 1

template_path = "./templates"
dataset_file_path = 'ARC path'
know_filter_url = "Know-filter path"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'vicuna'
if model_name == 'vicuna':
    base_model_url = "Inference model path"
    template_name = 'vicuna'
else:
    base_model_url = "Inference model path"
    template_name = ''
tokenizer = LlamaTokenizer.from_pretrained(base_model_url, device=device)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = 0
model = LlamaForCausalLM.from_pretrained(base_model_url, torch_dtype=torch.bfloat16).eval().to(device)

model.config.use_cache = False

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

know_filter_tokenizer = AutoTokenizer.from_pretrained(know_filter_url, device=device)

class Env:
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.model.generation_config.temperature=1
        self.model.generation_config.top_p=0.9
        self.model.generation_config.repetition_penalty=1.2
        self.model.generation_config.max_new_tokens=256
        self.guide_list = [[29889], [29991], [29973]]
        
    def _logits_processor(self, config, input_length):
        """Set up logits processor based on the generation config."""
        processor = LogitsProcessorList()

        if (
            config.min_new_tokens is not None
            and config.min_new_tokens > 0
            and config.eos_token_id is not None
        ):
            processor.append(
                MinNewTokensLengthLogitsProcessor(
                    prompt_length_to_skip=input_length,
                    min_new_tokens=config.min_new_tokens,
                    eos_token_id=config.eos_token_id,
                )
            )

        if (
            config.temperature is not None
            and config.temperature > 0
            and config.temperature != 1.0
        ):
            processor.append(TemperatureLogitsWarper(config.temperature))

        if config.top_p is not None and config.top_p > 0 and config.top_p < 1:
            processor.append(TopPLogitsWarper(config.top_p))

        return processor    
    
    @torch.inference_mode()
    def generate(self, prompts, guide, mult_reward=None, memory = None):
        config = self.model.generation_config
        pad_token_id = config.pad_token_id
        bos_token_id = config.bos_token_id
        eos_token_id = config.eos_token_id
        
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id[0]
        
        encodings = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        batch_size = len(prompts)
        input_length = encodings['input_ids'].shape[-1]
        
        input_ids = encodings['input_ids']
        
        processor = self._logits_processor(config, input_length)
        
        all_unfinished = input_ids.new_ones(batch_size)
        sentence_unfinished = all_unfinished.clone()
        
        pre_tokens = None
        
        while True:
            inputs = self.model.prepare_inputs_for_generation(
                input_ids, logprobs=0,
                min_new_tokens=0,
                max_new_tokens=256,
                temperature=config.temperature,
                top_p=config.top_p,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True
            ) 
            outputs = self.model(**inputs, return_dict=True, output_attentions=False, output_hidden_states=False)
            logits = outputs.logits[:, -1, :]
            logits = processor(input_ids, logits)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            if mult_reward is not None:
                reward_list = []
                reward_list.append(probs[0][mult_reward[0]])     # [batchsize, num_labels]?
                return reward_list
                        
            if (config.top_p is not None and config.top_p <= 0) or (
                config.temperature is not None and config.temperature <= 0
            ):
                tokens = torch.argmax(probs, dim=-1)[:, None]
            else:
                tokens = torch.multinomial(probs, num_samples=1)
            tokens = tokens.squeeze(1)

            if eos_token_id is not None:
                not_eos = sum(tokens != i for i in eos_token_id)
                all_unfinished = all_unfinished.mul(not_eos.long())
                sentence_unfinished = sentence_unfinished.mul(not_eos.long())
                
            if guide and self.guide_list is not None and pre_tokens is not None:
                for k in range(len(self.guide_list)):
                    not_ok = sum(pre_tokens != i for i in self.guide_list[k])
                    sentence_unfinished = sentence_unfinished.mul(not_ok.long())
                    
            if pad_token_id is not None:
                tokens = tokens * all_unfinished + pad_token_id * (1 - all_unfinished)
                tokens = tokens * sentence_unfinished + pad_token_id * (1 - sentence_unfinished)

            input_ids = torch.cat([input_ids, tokens[:, None]], dim=-1)
            pre_tokens = tokens
                
            status = sentence_unfinished.clone()
            if input_ids.shape[-1] >= config.max_new_tokens:
                status = 0 - status
                all_unfinished.fill_(0)  
            if status.max() <= 0:
                break
        return input_ids, all_unfinished

    def step(self, prompts, reward_prompts, labels, temperature=1, top_p=0.9, memory=None):
        self.model.generation_config.temperature = temperature
        self.model.generation_config.top_p = top_p
        generated_id, all_unfinished = self.generate(
            prompts=prompts,
            guide=True,
            memory=memory
        )
        decoded = self.tokenizer.batch_decode(generated_id, skip_special_tokens=True)
        knows = prompter.get_response(decoded)
        reward_prompts = prompter.replace_know(prompt=reward_prompts, knowledge=knows)
        rewards = self.get_reward(prompts=reward_prompts, labels=labels)
        ret_reward = 2 * rewards[0] * 100 - memory.pre_reward
        memory.pre_reward = rewards[0] * 100

        del generated_id
        torch.cuda.empty_cache()
        return decoded, ret_reward, all_unfinished[0]

    def guide_gen(self, prompts, reward_prompts, labels, temperature=1, top_p=0.9):
        self.model.generation_config.temperature = temperature
        self.model.generation_config.top_p = top_p
        generated_id, all_unfinished = self.generate(
            prompts=prompts,
            guide=True
        )
        decoded = self.tokenizer.batch_decode(generated_id, skip_special_tokens=True)
        knows = prompter.get_response(decoded)
        reward_prompts = prompter.replace_know(prompt=reward_prompts, knowledge=knows)
        rewards = self.get_reward(prompts=reward_prompts, labels=labels)
        
        del generated_id
        torch.cuda.empty_cache()
        return decoded, rewards, all_unfinished[0]
        
    def get_reward(self, prompts, labels, temperature=1, top_p=1):
        self.model.generation_config.temperature = temperature
        self.model.generation_config.top_p = top_p
        rewards = self.generate(
            prompts=prompts,
            guide=False,
            mult_reward=self.label2inputid(labels)
        )
        return rewards
    
    def label2inputid(self, labels):
        temp = []
        labels2ids={'A':29909,'B':29933,'C':29907,'D':29928}
        temp.append(labels2ids[labels[0]])
        return temp
        
import json
import re
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = osp.join(template_path, f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        if type(output) == str:
            output = output.split(self.template["response_split"])[1].strip()
        if type(output) == list:
            for i in range(len(output)):
                output[i] = output[i].split(self.template["response_split"])[1].strip()
        return output
    
    def replace_know(self, prompt, knowledge):
        if type(prompt) == str:
            return prompt.replace('[replace_here]', knowledge)
        if type(prompt) == list:
            temp = []
            for i in range(len(prompt)):
                if type(knowledge) == list:
                    temp.append(prompt[i].replace('[replace_here]', knowledge[i]))
                else:
                    temp.append(prompt[i].replace('[replace_here]', knowledge))
            return temp
        return prompt
    
    def get_question(self, prompt):
        pat = re.compile('(?<=Question:\n).*?(?=\n)')
        res = pat.search(prompt).group()
        return res
    
prompter = Prompter(template_name)

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CommenSenseDataset(Dataset):
    def __init__(self, data_file, split):
        self.data = load_dataset(data_file, split=split)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

data_dev = load_dataset(dataset_file_path,'ARC-Challenge', split='test')

def cs_knowledge_collote_fn(batch_samples):
    batch_data = []
    batch_gen_know = []
    batch_question = []
    batch_choice = []
    batch_label = []
    digit2char = {'1':'A','2':'B','3':'C','4':'D'}
    for sample in batch_samples:
        t = ''
        if sample['choices']['label'][0] in ['1','2','3','4']: 
            for i in range(len(sample['choices']['label'])):
                sample['choices']['label'][i] = digit2char[str(sample['choices']['label'][i])]
            sample['answerKey'] = digit2char[sample['answerKey']]
        for i,j in zip(sample['choices']['label'],sample['choices']['text']):
            t = t + '('+str(i) +') '+str(j) +' '
            
        if template_name == 'vicuna' :
            p = prompter.generate_prompt(instruction='Choose the correct answer to the question based on knowledge. Knowledge: [replace_here] Question: '+sample['question']+' Answer Choices: '+t.strip())+' ('
            k = prompter.generate_prompt(instruction='Provide some knowledge related to the question and no less than 50 words. Question: '+sample['question'])
        else:
            p = prompter.generate_prompt(instruction='Choose the correct answer to the question based on knowledge.', input='Knowledge:\n[replace_here]\nQuestion:\n'+sample['question']+'\nAnswer Choices:\n'+t.strip())+'('
            k = prompter.generate_prompt(instruction='Provide some knowledge related to the question and no less than 50 words.',input='Question:\n'+sample['question'])
        
        batch_gen_know.append(k)
        batch_data.append(p)
        batch_choice.append(t.strip())
        batch_question.append(sample['question'])
        batch_label.append(sample['answerKey'])
    return batch_data, batch_gen_know, batch_question, batch_choice, batch_label

def compare_label(preds, labels):
    if len(preds) != len(labels):
        raise ValueError('number error')
    res = []
    for i in range(len(preds)):
        if preds[i][1] == labels[i]:
            res.append(True)
        else:
            res.append(False)
    return res

know_filter_model = MonoT5(know_filter_url, token_false='▁false', token_true = '▁true')

print(f'Inference model：{template_name}')
print(f'Know-filter size：'+str(sum(p.numel() for p in know_filter_model.model.parameters())))

def search_(docid, rank_list):
    know, pred = '', ''
    for i in rank_list:
        if i[0] == docid:
            know, pred = i[1], i[2]
    if know == '' or pred == '':
        print('search fail')
    return know, pred
        
import time
@torch.inference_mode()
def test():
    env = Env(model, tokenizer)
    print('-'*20+'start'+'-'*20)
    dev_dataloader = DataLoader(data_dev, batch_size=dev_batch_size, shuffle=False, collate_fn=cs_knowledge_collote_fn, num_workers=4, drop_last=True)
    start_time = time.time()
    config = env.model.generation_config
    config.temperature = 1
    config.top_p = 0.9
    a_config = GenerationConfig(

            max_new_tokens=20,
            do_sample=False,
        )
    q_config = GenerationConfig(

            max_new_tokens=256,
            do_sample=False,
        )
    acc_count = 0
    pred = ''
    for idx, (batch_data, batch_gen_know, batch_question, batch_choice, batch_label) in enumerate(dev_dataloader):
        state = [batch_question[0].replace('  ',' '),'']
        done = 1
        query = Query(state[0])
        while done == 1:
            combined_prompts = [batch_gen_know[0] + state[1]]
            threshold = 10  
            
            rank_list = [] 
            while threshold > 0:
                decoded, reward, done = env.guide_gen(prompts=combined_prompts, reward_prompts=batch_data, labels=batch_label, temperature=1)
                reward_prompts = prompter.replace_know(prompt=batch_data, knowledge=decoded[0])
                encodings = tokenizer(reward_prompts, padding=True, return_tensors="pt").to(device)
                generated_ids = env.model.generate(**encodings, generation_config=a_config, max_new_tokens=20)
                preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                preds = prompter.get_response(preds)
                
                rank_list.append([reward[0].item(), decoded[0], preds[0]])
                threshold -= 1
            texts = [ Text('answer: '+ p[2][4:].lower().strip('.') + '. explanation: ' + p[1], {'docid': p[0]}, 0) for p in rank_list]
            reranked = know_filter_model.rerank(query, texts)
                
            # -----------------------fusing-------------------------
            know1, pred = search_(reranked[0].metadata["docid"], rank_list)
            sorted_list = [know1,]
            n = 2
            for i in range(1,n):
                know_, pred_ = search_(reranked[i].metadata["docid"], rank_list)
                sorted_list.append(know_)
            
            re_error = False
            inputs_know = ''
            if n>1 and sorted_list[0] != sorted_list[-1] and sorted_list[0] != state[1] and sorted_list[-1] != state[1]: 
                for i in range(n):
                    if state[1] != '':  
                        match_res = sorted_list[i].replace(state[1],'').strip()
                        if match_res == None or match_res == '':
                            re_error = True
                            state[1] = sorted_list[0]
                            break
                        else:
                            inputs_know+=' Knowledge: '+match_res
                    else:
                        inputs_know+=' Knowledge: '+sorted_list[i]
                if re_error==False :
                    fusion_prompts = prompter.generate_prompt(instruction='Rewriting the given knowledge into a new sentence requires retaining the part of the given knowledge that is relevant to the question and correct.'+inputs_know+' Question: '+batch_question[0])
                    encodings = tokenizer(fusion_prompts, padding=True, return_tensors="pt").to(device)
                    generated_ids = env.model.generate(**encodings, generation_config=q_config, max_new_tokens=256)
                    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    fusion_res = prompter.get_response(decoded)
                    if state[1] != '':
                        fusion_res[0] = state[1] + ' ' + fusion_res[0]
                    reward_prompts = prompter.replace_know(prompt=batch_data, knowledge=fusion_res[0])
                    fusion_reward = env.get_reward(prompts=reward_prompts, labels=batch_label)
                    encodings = tokenizer(reward_prompts, padding=True, return_tensors="pt").to(device)
                    generated_ids = env.model.generate(**encodings, generation_config=a_config, max_new_tokens=20)
                    preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    preds = prompter.get_response(preds)

                    texts = [ Text('answer: '+ preds[0][4:].lower().strip('.') + '. explanation: ' + fusion_res[0], {'docid': fusion_reward[0].item()}, 0)]
                    fusion_reranked = know_filter_model.rerank(query, texts)
                    if fusion_reranked[0].score > reranked[0].score:
                        state[1] = fusion_res[0]
                        pred = preds[0]
                    else:
                        state[1] = sorted_list[0]
            else:  
                state[1] = sorted_list[0]
            
        res = compare_label([pred], batch_label)
        acc_count += sum(res)
        print(f'idx: {idx}, acc_count: {acc_count}, acc: {acc_count/(idx+1)}')
    end_time = time.time()
    print('-'*20+'over, acc:'+str(acc_count/len(dev_dataloader)))
        
test()