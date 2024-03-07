import warnings
warnings.filterwarnings("ignore")

import torch

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

def change_target(x):
    if 'positive' in x or 'Positive' in x:
        return 'positive'
    elif 'negative' in x or 'Negative' in x:
        return 'negative'
    else:
        return 'neutral'

def process_batch(model, tokenizer, tmp_context, out_text_list):
    tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512)
    for k in tokens.keys():
        tokens[k] = tokens[k].cuda()
    res = model.generate(**tokens, max_length=512)
    res_sentences = [tokenizer.decode(i) for i in res]
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
    out_text_list += out_text
    torch.cuda.empty_cache()

    return out_text_list


def process_batch_zeroshot_cot(model, tokenizer, tmp_context, out_text_list):
    tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512)
    for k in tokens.keys():
        tokens[k] = tokens[k].cuda()
    res = model.generate(**tokens, max_length=512)
    res_sentences = [tokenizer.decode(i) for i in res]
    out_text = [o.split("Answer: ")[1] for o in res_sentences]
    out_text_list += out_text
    torch.cuda.empty_cache()

    return out_text_list


def add_instructions_default():
    pass

def add_instructions_fiqa():
    pass

def add_instructions_zeroshot_cot():
    pass

