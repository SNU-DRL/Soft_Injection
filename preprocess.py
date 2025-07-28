#### 
import json
import random
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L15
### but fixed an error that occurred when the output started with an integer (e.g., for 'april: 5', the next token after ':' is a space character (' '), which interfered with training the soft head-selection parameters).
def word_pairs_to_prompt_data(word_pairs : dict,
                              instructions: str = "",
                              prefixes: dict = {"input":"Q:", "output":"A:","instructions":""},
                              separators: dict = {"input":"\n", "output":"\n\n", "instructions":""},
                              query_target_pair: dict = None, prepend_bos_token=False,
                              shuffle_labels=False, prepend_space=True) -> dict:
    """Takes a dataset of word pairs, and constructs a prompt_data dict with additional information to construct an ICL prompt.
    Parameters:
    word_pairs: dict of the form {'word1':['a', 'b', ...], 'word2':['c', 'd', ...]}
    instructions: prefix instructions for an ICL prompt
    prefixes: dict of ICL prefixes that are prepended to inputs, outputs and instructions
    separators: dict of ICL separators that are appended to inputs, outputs and instructions
    query_target_pair: dict with a single input-output pair acting as the query for the prompt
    prepend_bos_token: whether or not to prepend a BOS token to the prompt
    shuffle_labels: whether to shuffle the ICL labels
    prepend_space: whether to prepend a space to every input and output token 

    Note 1) Without the space after the colon (i.e., ":{text}"), ":" sometimes does not get tokenized separately by the tokenizer.
            Therefore, we prepend a space after the colon in the input/output pairs. (This is a default choice from Function Vectors)
    Note 2) For ": {numeric outputs}" (e.g., ": 5"), the next token of ":" is " " (only the space). Therefore, we skip the space for numeric outputs.
            This is different from the default choice in Function Vectors, where they missed this edge case in some datasets (word_length, squad_val, etc.).

    Returns: 
    prompt_data: dict containing ICL prompt examples, and template information
    """
    prompt_data = {}
    prompt_data['instructions'] = instructions
    prompt_data['separators'] = separators
    if prepend_bos_token:
        prefixes = {k:(v if k !='instructions' else '<|endoftext|>' + v) for (k,v) in prefixes.items()}
    prompt_data['prefixes'] = prefixes

    if query_target_pair is not None:
        query_target_pair = {k:(v[0] if isinstance(v, list) else v) for k,v in query_target_pair.items()}
    prompt_data['query_target'] = query_target_pair
        
    if shuffle_labels:
        randomized_pairs = [np.random.permutation(x).tolist() if i==1 else x for (i,x) in enumerate(list(word_pairs.values()))] # shuffle labels only
        if prepend_space:
            # prompt_data['examples'] = [{'input':' ' + w1, 'output':' ' + w2} for (w1,w2) in list(zip(*randomized_pairs))]
            # prompt_data['query_target'] = {k:' ' + v for k,v in query_target_pair.items()} if query_target_pair is not None else None
            prompt_data['examples'] = [{'input':' ' + w1, 'output':' ' + str(w2)} for (w1,w2) in list(zip(*randomized_pairs))] # modified
            prompt_data['query_target'] = {k:' ' + str(v) for k,v in query_target_pair.items()} if query_target_pair is not None else None # modified
        else:
            prompt_data['examples'] = [{'input':w1, 'output':w2} for (w1,w2) in list(zip(*randomized_pairs))]
    else:    
        if prepend_space:
            # prompt_data['examples'] = [{'input':' ' + w1, 'output':' ' + str(w2)} for (w1,w2) in list(zip(*word_pairs.values()))]
            # prompt_data['query_target'] = {k:' ' + str(v) for k,v in query_target_pair.items()} if query_target_pair is not None else None
            prompt_data['examples'] = [
                {
                    'input': ' ' + w1,
                    'output': (' ' + str(w2)) if not str(w2).isdigit() else str(w2)
                }
                for (w1, w2) in list(zip(*word_pairs.values()))
            ]
            prompt_data['query_target'] = {
                k: (' ' + str(v)) if not str(v).isdigit() else str(v)
                for k, v in query_target_pair.items()
            } if query_target_pair is not None else None
        else:
            # prompt_data['examples'] = [{'input':w1, 'output':w2} for (w1,w2) in list(zip(*word_pairs.values()))]
            prompt_data['examples'] = [{'input':w1, 'output':str(w2)} for (w1,w2) in list(zip(*word_pairs.values()))]
            prompt_data['query_target'] = {k:str(v) for k,v in query_target_pair.items()} if query_target_pair is not None else None
    
    return prompt_data


def create_fewshot_primer(prompt_data) -> str:
    """Creates the primer string for GPT in-context learning
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information

    Returns:
    prompt: the constructed ICL prompt primer as a string
    """       
    prompt = ''
    prompt += prompt_data['prefixes']['instructions'] + prompt_data['instructions'] + prompt_data['separators']['instructions']
    
    for example in prompt_data['examples']:
        
        prompt += prompt_data['prefixes']['input'] + example['input'] + prompt_data['separators']['input']
        prompt += prompt_data['prefixes']['output'] + example['output'] + prompt_data['separators']['output']
        
    return prompt


def create_prompt(prompt_data, sentence=None) -> str:
    """Creates a prompt using the specified sentence for GPT in-context learning
    
    Parameters:
    prompt_data: dict containing ICL prompt examples, and template information
    sentence: a query string (sentence/word) to include in the ICL prompt

    Returns:
    prompt: the constructed ICL prompt as a string
    """
    if sentence is None and prompt_data['query_target'] is not None:
        sentence = prompt_data['query_target']['input']

    if isinstance(sentence, list):
        sentence = sentence[0]

    prompt_init = create_fewshot_primer(prompt_data)    
    prompt = prompt_init + prompt_data['prefixes']['input'] + sentence + prompt_data['separators']['input']
    prompt += prompt_data['prefixes']['output']
    
    return prompt 


def get_answer_id(query, answer, tokenizer):
    """
    Parameters:
    query (str): query as a string
    answer (str): expected answer as a string
    tokenizer: huggingface tokenizer
    
    Returns: 
    answer_ids (list): A list of the contextualized tokens of the answer
    """
    source = tokenizer(query, truncation=False, padding=False).input_ids
    target = tokenizer(query + answer, truncation=False, padding=False).input_ids
    if not len(source) < len(target) < tokenizer.model_max_length:
        print(f"Source: {source}, Target: {target}")
        print(f"Error! Source length: {len(source)}, Target length: {len(target)}, Model max length: {tokenizer.model_max_length}")
    assert len(source) < len(target) < tokenizer.model_max_length
    answer_ids = target[len(source): ]
    return answer_ids

###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L15
class ICLDataset:
    """
    A simple dataset class containing input-output pairs, used for ICL prompt construction.
    """
    def __init__(self, dataset):    
        if isinstance(dataset, str):
            self.raw_data = pd.read_json(dataset)
        elif isinstance(dataset, dict):
            self.raw_data = pd.DataFrame(dataset)
        self.raw_data = self.raw_data[['input', 'output']]

    def __getitem__(self,i):       
        if isinstance(i, int):
            return self.raw_data.iloc[i].to_dict()
        elif isinstance(i, slice):
            return self.raw_data.iloc[i].to_dict(orient='list')
        elif isinstance(i, list) or isinstance(i, np.ndarray):            
            return self.raw_data.iloc[i].to_dict(orient='list')
        elif isinstance(i, str):
            if i not in self.raw_data.columns:
                raise KeyError(f"Column '{i}' not in the dataset. Current columns in the dataset: {self.raw_data.columns.to_list()}")
            else:
                return self.raw_data[i].to_list()
        else:
            raise ValueError(f"{i} is not a valid index type. Expected one of: [int, list, np.ndarray, slice, str]")

    def __len__(self):
        return len(self.raw_data)
    
    def __repr__(self):
        s = "ICLDataset" + "({\n\tfeatures: " + f"{self.raw_data.columns.to_list()},\n\tnum_rows: {self.__len__()}" + "\n})"
        return s


def load_dataset(data_path: str = None):
    dataset = ICLDataset(data_path)
    return dataset


def get_format_func(cur_dataset):

    if cur_dataset == "icl":
        return format_icl


def format_icl(all_data, cur_items=None, num_shot=0, model_helper=None, batch_size=1, dataset_names=None):
    """
    all_data: list of tuples of datasets
      ex) [(train1, valid1), (train2, valid2), …, (train10, valid10)]
    cur_items: list of current items to use for ICL last query
      
    Returns:
     - tokenized_prompt: tokenized full prompt
     - image_list: we do not use images in ICL, so this is None
     - target_token_id: the list of target token id (only the first token of the target word)
     - dataset_idx: the index of the randomly chosen dataset from all_data
    """
    prompts = []
    target_token_ids = []

    if all_data is not None:
        dataset_idx, train_set = random.choice(list(enumerate(all_data)))
    else:
        dataset_idx = None
    
    if dataset_names is not None and dataset_idx is not None:
        if dataset_names[dataset_idx] == "squad_val" and batch_size > 2:
            batch_size = 2
            print(f"Warning: batch size is set to 2 for {dataset_names[dataset_idx]} dataset to avoid memory overflow.")

    if cur_items is None:
        cur_items = [None] * batch_size
    
    for i in range(batch_size):    

        if num_shot == 0:
            word_pairs = {'input':[], 'output':[]}
            if cur_items[i] is None:
                random_indices = random.sample(range(len(train_set)), 1)
                cur_item = train_set[random_indices[-1]]
            else:
                cur_item = cur_items[i]
        else:
            random_indices = random.sample(range(len(train_set)), num_shot + 1)
            subset = [train_set[i] for i in random_indices[:-1]]
            word_pairs = {}
            for item in subset:
                for key, value in item.items():
                    word_pairs.setdefault(key, []).append(value)
            if cur_items[i] is None:
                cur_item = train_set[random_indices[-1]]
            else:
                cur_item = cur_items[i]
        
        word_pairs_test = cur_item
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, shuffle_labels=False)
        query = prompt_data['query_target']['input']
        if isinstance(query, list):
            query = query[0]
        prompt_string = create_prompt(prompt_data=prompt_data, sentence=query)
        prompts.append(prompt_string)
        
        target = prompt_data['query_target']['output']
        target = target[0] if isinstance(target, list) else target
        target_token_id = get_answer_id(create_prompt(prompt_data), target, model_helper.tokenizer)[:1]
        target_token_ids.append(target_token_id)

    tokenized_prompt = model_helper.tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
    image_list = None # Used only for vision-language tasks only
    
    return tokenized_prompt, image_list, target_token_ids, dataset_idx, target


