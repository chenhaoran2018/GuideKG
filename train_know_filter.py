import os
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from pygaggle.rerank.transformer import MonoT5
from pygaggle.rerank.base import Query, Text
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback
)

class MonoT5Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = f'Query: {sample[0]} Document: {sample[1]} Relevant:'
        return {
          'text': text,
          'labels': sample[2],
          'score_labels': sample[3],
        }
    
def custom_validation_function(model, eval_dataset, eval_dataset_2, n=1):
    acc_num = 0
    for exm in eval_dataset:
        query = Query(exm['question'])
        texts = [ Text('answer: '+ p[2] + '. explanation: ' + p[0], {'docid': p[1]}, 0) for p in exm['knowledge']]
        reranked = model.rerank(query, texts)

        flag_1 = False
        for p in exm['knowledge']:
            if reranked[0].text == 'answer: '+ p[2] + '. explanation: ' + p[0] and p[1] > 0.5:
                flag_1 = True
                break

        gt = ['answer: '+ k[2] + '. explanation: ' + k[0] for k in exm['knowledge'][-n:]]
        pred = sorted([reranked[i].text for i in range(n)])
        if gt == pred or flag_1 == True:
            acc_num += 1

    print(f'csqa2:{acc_num/len(eval_dataset)}')
    
    acc_num_2 = 0
    for exm in eval_dataset_2:
        query = Query(exm['question'])
        texts = [ Text('answer: '+ p[2][4:] + '. explanation: ' + p[0], {'docid': p[1]}, 0) for p in exm['knowledge']]
        reranked = model.rerank(query, texts)

        flag_1 = False
        for p in exm['knowledge']:
            if reranked[0].text == 'answer: '+ p[2][4:] + '. explanation: ' + p[0] and p[1] > 0.5:
                flag_1 = True
                break

        gt = ['answer: '+ k[2][4:] + '. explanation: ' + k[0] for k in exm['knowledge'][-n:]]
        pred = sorted([reranked[i].text for i in range(n)])
        if gt == pred or flag_1 == True:
            acc_num_2 += 1
    print(f'csqa:{acc_num_2/len(eval_dataset_2)}')

    return acc_num/len(eval_dataset)

class ValidationCallback(TrainerCallback):
    def __init__(self, eval_dataset, eval_dataset_2, tokenizer):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.eval_dataset_2 = eval_dataset_2

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch > 1.0:
            reranker =  MonoT5(model=kwargs['model'].eval())
            evaluation_results = custom_validation_function(model=reranker, eval_dataset=self.eval_dataset, eval_dataset_2=self.eval_dataset_2)

            kwargs['model'].train()
            print("csqa2.0:", evaluation_results)

import jsonlines
eval_dataset = []
with jsonlines.open('./datsets/csqa2_vicuna_test.jsonl',mode='r') as r:
    for row in r:
        eval_dataset.append(row)

eval_dataset_2 = []
with jsonlines.open('./datsets/csqa_vicuna_test.jsonl',mode='r') as r:
    for row in r:
        eval_dataset_2.append(row)

def main():
    triples_path = ''
    output_model_path = ''
    logging_steps = 60
    per_device_train_batch_size = 32
    gradient_accumulation_steps = 4
    learning_rate = 3e-4
    epochs = 20

    save_every_n_steps = 0
    base_model = 'monot5-large-msmarco-10k'
    
    device = torch.device('cuda')
    torch.manual_seed(123)

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    valid_callback = ValidationCallback(eval_dataset, eval_dataset_2, tokenizer)

    import jsonlines


    train_samples = []
    with jsonlines.open('./datasets/vicuna_train.jsonl',mode='r') as r:
        for row in r:
            if row['pos_pred'][0] == '(':
                train_samples.append((row['question'], 'answer: '+ row['pos_pred'][4:] + '. explanation: ' + row['pos'], 'true', row['pos_score']))
                train_samples.append((row['question'], 'answer: '+ row['neg_pred'][4:] + '. explanation: ' + row['neg'], 'false', row['neg_score']))
            else:
                train_samples.append((row['question'], 'answer: '+ row['pos_pred'] + '. explanation: ' + row['pos'], 'true', row['pos_score']))
                train_samples.append((row['question'], 'answer: '+ row['neg_pred'] + '. explanation: ' + row['neg'], 'false', row['neg_score']))

    def smart_batching_collate_text_only(batch): 
        texts = [example['text'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']
        tokenized['score_labels'] = torch.tensor([example['score_labels'] for example in batch])

        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)
        return tokenized

    import random
    random.seed(3407)
    random.shuffle(train_samples)
    dataset_train = MonoT5Dataset(train_samples)

    if save_every_n_steps > 0:
        steps = save_every_n_steps
        strategy = 'steps'
    else:
        steps = 1
        strategy = 'epoch'

    train_args = Seq2SeqTrainingArguments(
        output_dir=output_model_path,
        do_train=True,
        save_strategy=strategy,
        save_steps =steps, 
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=5e-5,
        num_train_epochs=epochs,
        warmup_steps=1000,  
        adafactor=True,
        seed=1,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
        callbacks=[valid_callback]
    )

    trainer.add_callback(valid_callback)
    trainer.train()

    trainer.save_model(output_model_path)
    trainer.save_state()

if __name__ == "__main__":
    main()