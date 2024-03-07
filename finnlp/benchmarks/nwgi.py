import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch
from finnlp.benchmarks.utils import format_example, change_target, process_batch

dic = {
    'strong negative':"negative",
    'moderately negative':"negative",
    'mildly negative':"neutral",
    'strong positive':"positive",
    'moderately positive':"positive",
    'mildly positive':'neutral',
    'neutral':'neutral',
}

def test_nwgi(model, tokenizer, batch_size = 8, prompt_fun = None, processor = process_batch):
    dataset = datasets.load_dataset('oliverwang15/news_with_gpt_instructions')
    dataset = dataset['test'].to_pandas()
    dataset['output'] = dataset['label'].apply(lambda x:dic[x])

    if prompt_fun is None:
        dataset["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        dataset["instruction"] = dataset.apply(prompt_fun, axis = 1)
    dataset["input"] = dataset["news"]

    dataset = dataset[['input', 'output', 'instruction']]
    dataset[["context","target"]] = dataset.apply(format_example, axis = 1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{dataset['context'][0]}\n\n")

    context = dataset['context'].tolist()
    
    total_steps = dataset.shape[0]//batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size:(i+1)* batch_size]
        out_text_list = processor(model, tokenizer, tmp_context, out_text_list) 

    dataset["out_text"] = out_text_list
    dataset["new_target"] = dataset["target"].apply(change_target)
    dataset["new_out"] = dataset["out_text"].apply(change_target)

    acc = accuracy_score(dataset["new_target"], dataset["new_out"])
    f1_macro = f1_score(dataset["new_target"], dataset["new_out"], average = "macro")
    f1_micro = f1_score(dataset["new_target"], dataset["new_out"], average = "micro")
    f1_weighted = f1_score(dataset["new_target"], dataset["new_out"], average = "weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return dataset
