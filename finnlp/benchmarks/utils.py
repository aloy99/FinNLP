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

def zeroshot_cot_process_factory(answer_trigger):
    def process_batch_zeroshot_cot(model, tokenizer, tmp_context, out_text_list):
        tokens = tokenizer(tmp_context, return_tensors='pt', padding=True, max_length=512)
        for k in tokens.keys():
            tokens[k] = tokens[k].cuda()
        z_sentences = model.generate(**tokens, max_length=512)
        z_sentences = [tokenizer.decode(i) for i in z_sentences]
        z2_sentences = [i + " " + answer_trigger for i in z_sentences]

        z2_tokens = tokenizer(z2_sentences, return_tensors='pt', padding=True, max_length=768)
        torch.cuda.empty_cache()
        for k in z2_tokens.keys():
            z2_tokens[k] = z2_tokens[k].cuda()

        res = model.generate(**z2_tokens, max_length=768)
        res_sentences = [tokenizer.decode(i) for i in res]
        out_text = [o.split(answer_trigger)[1] for o in res_sentences]
        out_text_list += out_text
        torch.cuda.empty_cache()

        return out_text_list
    return process_batch_zeroshot_cot


def add_instructions_default():
    pass

def add_instructions_fiqa():
    pass

def add_instructions_zeroshot_cot():
    pass

