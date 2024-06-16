import sys
import copy
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
    RobertaModel,
    GenerationConfig, LlamaTokenizer, LlamaForCausalLM
)

from transformers import logging
logging.set_verbosity_error()

max_epoch = 4
batch_size = 1
dev_batch_size = 1
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'vicuna'
if model_name == 'vicuna':
    base_model_url = "/home/kai/alpaca-lora/weights/vicuna-7b-v1.5"
    template_name = 'vicuna'
else:
    base_model_url = "/home/kai/alpaca-lora/weights/alpaca-7b-complete"
    template_name = ''
tokenizer = LlamaTokenizer.from_pretrained(base_model_url, device=device)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = 0
model = LlamaForCausalLM.from_pretrained(base_model_url, torch_dtype=torch.float16).to(device)

model.config.use_cache = False


class Env:
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.model.generation_config.temperature=0.9
        self.model.generation_config.top_p=0.9
        self.model.generation_config.repetition_penalty=1.2
        self.model.generation_config.max_new_tokens=512
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
    def generate(self, prompts, guide, mult_reward=None):
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
                max_new_tokens=512,
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
                reward_list.append(probs[0][mult_reward[0]])
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
        
    def step(self, prompts, reward_prompts, labels, temperature=0.01, top_p=0.9):
        self.model.generation_config.temperature = temperature
        self.model.generation_config.top_p = top_p
        generated_id, all_unfinished = self.generate(
            prompts=prompts,
            guide=True,
        )
        decoded = self.tokenizer.batch_decode(generated_id, skip_special_tokens=True)
        knows = prompter.get_response(decoded)
        reward_prompts = prompter.replace_know(prompt=reward_prompts, knowledge=knows)
        rewards = self.get_reward(prompts=reward_prompts, labels=labels)
        
        #---------------------------------------------------------

        del generated_id
        torch.cuda.empty_cache()
        return decoded, ret_reward, all_unfinished[0], rewards[0]
        #----------------------------------------------------------
        
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
        labels2ids={'A':29909,'B':29933,'C':29907,'D':29928, 'E':29923}
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
        file_name = osp.join("/home/kai/alpaca-lora/templates", f"{template_name}.json")
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

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class CommenSenseDataset(Dataset):
    def __init__(self, data_file, split):
        self.data = load_dataset(data_file, split=split)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
#------------------------------------------------------------------------------------
data_train = load_dataset("/home/kai/alpaca-lora/data/commonsense_qa", split='train') 
#------------------------------------------------------------------------------------

def cs_knowledge_collote_fn(batch_samples):
    batch_data = []
    batch_gen_know = []
    batch_question = []
    batch_choice = []
    batch_label = []
    for sample in batch_samples:
        t = ''
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

import jsonlines

def cut_dig(number):
    res = 0.0
    if number<0.1:
        res = 0
    elif number >= 0.1 and number < 0.2:
        res = 0.1
    elif number >= 0.2 and number < 0.3:
        res = 0.2
    elif number >= 0.3 and number < 0.4:
        res = 0.3
    elif number >= 0.4 and number < 0.5:
        res = 0.4
    elif number >= 0.5 and number < 0.6:
        res = 0.5
    elif number >= 0.6 and number < 0.7:
        res = 0.6
    elif number >= 0.7 and number < 0.8:
        res = 0.7
    elif number >= 0.8 and number < 0.9:
        res = 0.8
    elif number >= 0.9 and number <= 1:
        res = 0.9
    return res
        
def main():
    a_config = GenerationConfig(
            max_new_tokens=10,
            do_sample=False,
        )

    jsonl_file = "gen_csqa_vicuna_train_2.jsonl"
    with jsonlines.open(jsonl_file, mode='w') as writer:
        for epoch in range(max_epoch):
            dev_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=cs_knowledge_collote_fn, num_workers=4)
            for idx, (batch_data, batch_gen_know, batch_question, batch_choice, batch_label) in enumerate(dev_dataloader):
                temperature = 1
                fanwei_count = {0:0, 0.1:0, 0.2:0, 0.3:0, 0.4:0, 0.5:0, 0.6:0, 0.7:0, 0.8:0, 0.9:0}
                done = 1
                state = [batch_question[0],'']
                data_record = {'question': batch_question[0]}
                while done == 1:
                    combined_prompts = [batch_gen_know[0] + state[1]]
                    decoded, reward, done, raw_reward = env.step(prompts=combined_prompts, reward_prompts=batch_data, temperature=temperature, labels=batch_label)
                    state[1] = decoded[0]
                    
                    reward_prompts = prompter.replace_know(prompt=batch_data, knowledge=decoded[0])
                    encodings = tokenizer(reward_prompts, padding=True, return_tensors="pt").to(device)
                    generated_ids = env.model.generate(**encodings, generation_config=a_config)
                    preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    preds = prompter.get_response(preds)
                    
                    data_record = {'question': batch_question[0], 'knowledge': decoded[0], 'choice': batch_choice[0].strip(), 'score': raw_reward.item(), 'pred': preds[0]}
                    writer.write(data_record)
                    fanwei_count[cut_dig(raw_reward.item())] += 1
                    data_len += 1
                    if idx == 0:
                        print(str(data_record))

                print('epoch {} idx {} data_len {} fanwei_count {}'.format(epoch, idx, data_len, str(fanwei_count)))
                
            
if __name__ == '__main__':
    main()