import json
import os
from tqdm import tqdm
import numpy as np
from concurrent import futures
# import tokenis
from underthesea import word_tokenize, ner
from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize

TOK = None

from multiprocessing import Queue
from queue import Empty

def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)

def load(file_name, is_test: bool = False):
    data = json.load(open(file_name, 'r', encoding='utf8'))
    output = {'ids': [], 'claims': [],
              'contexts': [], 'labels': []}
    for id in data.keys():
        output['ids'].append(id)
        output['contexts'].append(data[id]["context"])
        output['claims'].append(data[id]["claim"])
        output['labels'].append(data[id]["verdict"] if not is_test else None)

    return output

def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'pos': tokens.pos(),
        'ner': tokens.entities(),
    }
    return output

def preprocess_item(idx, claim, context, verdict:str = None):
    q_token = ner(claim)
    q_token = np.transpose(q_token)
    q_token = [list(_) for _ in q_token]
    #Tokenize token in claim
    q_word, q_pos, q_ner = q_token[0], q_token[1], q_token[3]


    c_token = ner(context)
    c_token = np.transpose(c_token)
    c_token = [list(_) for _ in c_token]
    
    c_word, c_pos, c_ner = c_token[0], c_token[1], c_token[3]
    if verdict is not None:
        return dict(
            id = idx,
            claim= q_word,
            claim_pos= q_pos,
            claim_ner= q_ner,
            c_document= c_word,
            c_pos= c_pos,
            c_ner= c_ner,
            label= verdict
        )
    return dict(
        id = idx,
        claim= q_word,
        claim_pos= q_pos,
        claim_ner= q_ner,
        c_document= c_word,
        c_pos= c_pos,
        c_ner= c_ner,
    )

def ner_pos(claim: str):
    """
    Return pos before ner
    """
    token = ner(claim)
    token = np.transpose(token)
    token = [list(_) for _ in token]
    
    return token[1], token[3]

def preprocess(file_name, output_name, is_test:bool = False):
    print(file_name)

    data = json.load(open(file_name, 'r', encoding='utf8'))

    examples = {}

    for k in data.keys():
        item = preprocess_item(k,
                                data[k]['claim'],
                                data[k]['evidence_predict'],
                                data[k]['verdict'])
        examples[k] = item
        

    
    print('Begin write %s'%(output_name))
    with open(output_name, "w", encoding='utf8') as outfile:
        json.dump(examples, outfile)

    print('End write %s'%(output_name))

def add_pos_ner4claim(data_file_name, preprocessed_file_name, output_name, is_test:bool = False):
    print(data_file_name)

    data = load(data_file_name, is_test)
    examples = json.load(open(preprocessed_file_name, 'r', encoding='utf8'))

    examples['h_ners'] = []
    examples['h_poses'] = []
    
    for idx in tqdm(range(len(examples['ids']))):
        h_pos, h_ner = ner_pos(data['claims'][idx])
        examples['h_ners'].append(h_pos)
        examples['h_poses'].append(h_ner)
    
    print('Begin write %s'%(output_name))
    with open(output_name, "w", encoding='utf8') as outfile:
        json.dump(examples, outfile)

    print('End write %s'%(output_name))

if __name__ == '__main__':
    #preprocess_ch('CQA_data/train-v1.1.json', 'train.json')
    #preprocess_ch('CQA_data/test-v1.1.json', 'test.json', True)

    # load_squad('UIT-ViQuAD 2.0/train.json')

    preprocess('ise-dsc01/train.json', 'ise-dsc01/data_preprocessed/train.json')
    # preprocess('ise-dsc01/ise-dsc01-public-test-offcial.json', 'ise-dsc01/data_preprocessed/public_test.json', True)

    # add_pos_ner4claim('ise-dsc01/ise-dsc01-train.json', 'ise-dsc01/data_preprocessed/train.json', 'ise-dsc01/data_preprocessed/train_claim_ner_pos.json')
    # add_pos_ner4claim('ise-dsc01/ise-dsc01-public-test-offcial.json', 'ise-dsc01/data_preprocessed/public_test.json', 'ise-dsc01/data_preprocessed/public_test_claim_ner_pos.json', True)