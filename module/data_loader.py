import json
import unicodedata
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm
from copy import deepcopy


PAD = 0
UNK = 1

def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def load_data(file_name):
    '''
    load train, test file
    Add other preprocessing?
    '''
    examples = json.load(open(file_name, 'r', encoding='utf8'))
    print(examples.keys())
    for idx in range(len(examples['ids'])):
        context = examples['contexts'][idx]
        claim = examples['claims'][idx]
        appear = []
        for w in context:
            if w in claim:
                appear.append(1)
            else:
                appear.append(0)
        examples['appears'].append(deepcopy(appear))
    return examples

def load_embedding(data_path, to_idx, embedding_size):
    '''
    Args:
        data_path: path to embedding file
        to_idx: dict, word --> index
    '''
    word_count = 0
    with open(data_path, 'r') as f:
        word_num, dim = f.readline().strip().split()
        word_num, dim = int(word_num), int(dim)
        init_w = np.random.uniform(-0.25,0.25,(len(to_idx), dim))

        for line in tqdm(f, desc='load embedding'):
            split_line = line.strip().split()
            word, vec = split_line[0], split_line[1:]
            if len(vec) != dim:
                # error due to unicode and we don't need it 
                continue
            if word in to_idx.keys():
                init_w[to_idx[word]] = np.array(list(map(float, vec)))
                word_count += 1
    print('Found pretrained embedding:', word_count)
    
    # PCA decomposition to reduce word2vec dimensionality
    # copy from other seq2seq code which works well but I never review it
    if embedding_size < dim:
        print('Reduce %d embedding dim to %d'%(dim, embedding_size))
        U, s, Vt = np.linalg.svd(init_w, full_matrices=False)
        S = np.zeros((dim, dim))
        S[:dim, :dim] = np.diag(s)
        init_w = np.dot(U[:, :embedding_size], S[:embedding_size, :embedding_size])

    return init_w

def split_exp(examples, ratio):
    
    split_num = int(len(examples) * ratio)
    return examples[:-split_num], examples[-split_num:]

def count_vocab(examples):
    vocab_count = Counter()
    max_context, max_q = 0, 0 
    for example in examples:
        context = example['context']
        q = example['q']
        vocab_count.update(context)
        vocab_count.update(q)

        max_context = max(max_context, len(context))
        max_q = max(max_q, len(q))

    return vocab_count, (max_context + 1, max_q)

class Vocabulary:
    def __init__(self, to_word, to_idx, to_pos, to_ner):
        self.to_word = to_word
        self.to_idx = to_idx
        self.to_pos = to_pos
        self.to_ner = to_ner
    
    def __len__(self):
        return len(self.to_word)
    
    def pos_size(self):
        return len(self.to_pos)
    
    def ner_size(self):
        return len(self.to_ner)
    
    def idx2word(self, idxs):
        words = []
        for idx in idxs:
            words.append(self.to_word[idx])
        return words
    
    def word2idx(self, words):
        idxs = []
        for word in words:
            idxs.append(self.to_idx['<UNK>'] if word not in self.to_idx.keys() else self.to_idx[word])
        return idxs
    
    def pos2idx(self, pos):
        idxs = []
        for tag in pos:
            idxs.append(self.to_pos['<UNK>'] if tag not in self.to_pos.keys() else self.to_pos[tag])
        return idxs
    
    def ner2idx(self, ner):
        idxs = []
        for tag in ner:
            idxs.append(self.to_ner['<UNK>'] if tag not in self.to_ner.keys() else self.to_ner[tag])
        return idxs
    
    def build_embedding(self, embed_file, wv_dim):
        vocab_size = self.__len__()
        emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))
        emb[0] = 0 # <PAD> should be all 0 (using broadcast)
    
        cnt = 0
        
        with open(embed_file, encoding="utf8") as f:
            
            for line in f:
                if cnt == 0:
                    cnt += 1
                    continue
                elems = line.split()
                token = normalize_text(''.join(elems[0:-wv_dim]))
                if token in self.to_idx:
                    emb[self.to_idx[token]] = [float(v) for v in elems[-wv_dim:]]
        return emb

def build_vocab(train, test, vocab_size):
    vocab_count, pad_lens = count_vocab(train + test)
    vocab = vocab_count.most_common()[:vocab_size]
    
    to_pos, to_ner = {}, {}
    to_pos['<PAD>'], to_pos['<UNK>'] = 0, 1
    to_ner['<PAD>'], to_ner['<UNK>'] = 0, 1
    for x in train + test:
        for tag in x['pos']:
            if tag not in to_pos.keys():
                to_pos[tag] = len(to_pos)
        
        for tag in x['ner']:
            if tag not in to_ner.keys():
                to_ner[tag] = len(to_ner)
    
    to_word, to_idx = {}, {}
    to_word[0], to_idx['<PAD>'] = '<PAD>', 0
    to_word[1], to_idx['<UNK>'] = '<UNK>', 1
    for w, c in vocab:
        to_word[len(to_word)] = w
        to_idx[w] = len(to_idx)
    return Vocabulary(to_word, to_idx, to_pos, to_ner), pad_lens

class DataEngine(Dataset):
    def __init__(self, datas, vocabulary, pad_lens):
        self.datas = datas
        self.vocabulary = vocabulary
        self.pad_context, self.pad_q = pad_lens
        
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.vectorize(self.datas['ids'][idx],
                              self.datas['claims'][idx],
                              self.datas['h_poses'][idx],
                              self.datas['h_ners'][idx],
                              self.datas['contexts'][idx],
                              self.datas['c_poses'][idx],
                              self.datas['c_ners'][idx],
                              self.datas['labels'][idx]
                              )
                              # '''-1 if len(self.datas[idx]['answers']) == 0 else '''
                              
                              # '''-1 if len(self.datas[idx]['answers']) == 0 else '''
                              

    def vectorize(self, context, context_pos, context_ner, q, ans_start, ans_end, appear):
        padding_context = ['<PAD>' for _ in range(self.pad_context - len(context))]
        context = context + padding_context
        context_pos = context_pos + padding_context
        context_ner = context_ner + padding_context

        padding_q = ['<PAD>' for _ in range(self.pad_q - len(q))]
        q = q + padding_q

        padding_appear = [0 for _ in range(self.pad_context - len(appear))]
        appear = appear + padding_appear

        context = torch.LongTensor(self.vocabulary.word2idx(context))
        context_pos = torch.LongTensor(self.vocabulary.pos2idx(context_pos))
        context_ner = torch.LongTensor(self.vocabulary.ner2idx(context_ner))
        q = torch.LongTensor(self.vocabulary.word2idx(q))
        ans_offset = torch.LongTensor([ans_start, ans_end])
        appear = torch.FloatTensor(appear)
        return context, context_pos, context_ner, q, ans_offset, appear
    
# if __name__ == "__main__":
#     train = load_data('/kaggle/input/squad1k/train_squad.json', False)
#     test = load_data('/kaggle/input/squad1k/dev_squad.json', False)[:50]
#     vocabulary, pad_lens = build_vocab(train, test, 25000)
    
#     Word_Vec_Dir = '/kaggle/input/glove-840b-300d/'
#     Embedding_File = Word_Vec_Dir + 'glove.840B.300d.txt'
    
#     print(vocabulary.build_embedding(Embedding_File, 300).shape)