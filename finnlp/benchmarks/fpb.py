import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score,f1_score
from datasets import load_dataset
from tqdm import tqdm
import datasets
import torch
from finnlp.benchmarks.utils import format_example, change_target, process_batch

dic = {
        0:"negative",
        1:'neutral',
        2:'positive',
    }

def test_fpb(model, tokenizer, batch_size = 8, prompt_fun = None, processor = process_batch):
    instructions = load_dataset("financial_phrasebank", "sentences_50agree")
    instructions = instructions["train"]
    instructions = instructions.train_test_split(seed = 42)['test']
    instructions = instructions.to_pandas()
    instructions.columns = ["input", "output"]
    instructions["output"] = instructions["output"].apply(lambda x:dic[x])

    if prompt_fun is None:
        instructions["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
    else:
        instructions["instruction"] = instructions.apply(prompt_fun, axis = 1)
    
    instructions[["context","target"]] = instructions.apply(format_example, axis = 1, result_type="expand")

    # print example
    print(f"\n\nPrompt example:\n{instructions['context'][0]}\n\n")


    context = instructions['context'].tolist()
    
    total_steps = instructions.shape[0]//batch_size + 1
    print(f"Total len: {len(context)}. Batchsize: {batch_size}. Total steps: {total_steps}")


    out_text_list = []
    for i in tqdm(range(total_steps)):
        tmp_context = context[i* batch_size:(i+1)* batch_size]
        out_text_list = processor(model, tokenizer, tmp_context, out_text_list) 

    instructions["out_text"] = out_text_list
    instructions["new_target"] = instructions["target"].apply(change_target)
    instructions["new_out"] = instructions["out_text"].apply(change_target)

    acc = accuracy_score(instructions["new_target"], instructions["new_out"])
    f1_macro = f1_score(instructions["new_target"], instructions["new_out"], average = "macro")
    f1_micro = f1_score(instructions["new_target"], instructions["new_out"], average = "micro")
    f1_weighted = f1_score(instructions["new_target"], instructions["new_out"], average = "weighted")

    print(f"Acc: {acc}. F1 macro: {f1_macro}. F1 micro: {f1_micro}. F1 weighted (BloombergGPT): {f1_weighted}. ")

    return instructions