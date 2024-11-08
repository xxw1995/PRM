from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import argparse
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from torch.nn import BCEWithLogitsLoss
from transformers import DataCollatorWithPadding, TrainerCallback
from datasets import concatenate_datasets
from prm.code.finetune_qwen import DATA_PATH


# 将输入数据进行预处理，主要是对文本进行分词，并生成标签
def preprocess_function(example):
    input = f"{example['question']} {example['process']}"
    tokenized_inputs = tokenizer(
        input,
        truncation=True,
        padding='max_length',
        max_length=2048,
    )
    length = len(tokenized_inputs['input_ids'])
    # step_tag_id 和 candidate_tokens 作为标记。
    indices = find_all_indices(tokenized_inputs['input_ids'], step_tag_id)
    if len(indices) != len(example['label']):
        example['label'] = example['label'][:len(indices)]
    assert len(indices) == len(example['label'])
    tokenized_inputs['labels'] = [-100] * length
    for i in range(len(indices)):
        if example['label'][i] == '+' or example['label'][i] == 1:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[0]
        elif example['label'][i] == '-' or example['label'][i] == 0:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[1]
        else:
            raise ValueError('label is wrong')
        tokenized_inputs['attention_mask'][indices[i]] = 0
    return tokenized_inputs

def find_all_indices(lst, element):
    return [i for i, x in enumerate(lst) if x == element]

def compute_metrics(eval_pred):
    pre, labels = eval_pred
    auc = roc_auc_score(pre[1], pre[0])
    ll = log_loss(pre[1], pre[0])
    acc = accuracy_score(pre[1], pre[0] > 0.5)
    result = {
        'auc': auc,
        'll': ll,
        'acc': acc,
    }
    print(result)
    return result

def preprocess_logits_for_metrics(logits, labels):
    labels_index = torch.argwhere(torch.bitwise_or(labels == candidate_tokens[0], labels == candidate_tokens[1]))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == candidate_tokens[1], 0, 1)
    logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [candidate_tokens[1], candidate_tokens[0]]]
    prob = torch.softmax(logits, dim=-1)
    return prob[:, 1], gold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--train_data_path", type=str, default="../../datasets/math_aps.json")
    parser.add_argument("--test_data_path", type=str, default="../../datasets/prm800k_test.json")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--total_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_eos_token=False)
    
    print(tokenizer.encode('a ки b'))
    print(tokenizer.encode('a b'))
    print(tokenizer.encode('a \n\n b'))
    print(tokenizer.encode('a b'))
    print(tokenizer.encode('a \n\n\n\n\n b'))
    print(tokenizer.encode('a b'))
    print(tokenizer.encode('a + b'))
    print(tokenizer.encode('a b'))
    print(tokenizer.encode('a - b'))
    print(tokenizer.encode('a b'))
    print(tokenizer.encode(' + -'))
    print(tokenizer.encode('+-'))
    print(tokenizer.eos_token_id)
    
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    
    candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}")
    print(candidate_tokens)
    step_tag_id = tokenizer.encode(f" {step_tag}")[-1]
    print('step_tag_id:', tokenizer.encode(f" {step_tag}"))
    print(tokenizer.encode(' \n\n\n\n\n')[-1])
    print('step_tag_id2:', tokenizer.encode(f"{step_tag2}"))

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    print(model.device)
    
    DATA_PATH['train'] = args.train_data_path
    DATA_PATH['test'] = args.test_data_path
    dataset = load_dataset('json', data_files=DATA_PATH)
    
    dataset['train'] = dataset['train'].select(range(10000))
    dataset['test'] = dataset['test'].select(range(10000))
    
    print('start processing')
    tokenized_datasets = dataset.map(preprocess_function)
    tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(['question', 'process', 'label'])
    tokenized_datasets['test'] = tokenized_datasets['test'].remove_columns(['question', 'process', 'label'])
    print('dataset processed')
    
    data_collator = DataCollatorWithPadding(tokenizer)
    
    BATCH_SIZE = args.total_batch_size
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.per_device_train_batch_size
    
    fp = f'bs_{args.total_batch_size}_lr_{args.learning_rate}'
    output_path = f'./prm_results_qwen/{fp}'
    
    class SaveBeforeEvaluateCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            print(111111)
            trainer.save_model(output_dir='./saved_model_before_eval')
    
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        callbacks=[SaveBeforeEvaluateCallback()],
    )
    
    trainer.train()
    
    model.save_pretrained('./fine_tuned_math_shepherd_lora_8bit')

if __name__ == "__main__":
    main()
