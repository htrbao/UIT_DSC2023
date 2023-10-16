import argparse
from module import FusionNet, DataEngine, load_data, load_embedding, split_exp, build_vocab
# from metrics import batch_score
# from embedding import load_embedding
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch import optim

class Arg():
  def __init__(self):
    self.rnn_type = 'lstm'
    self.hidden_size = 256
    self.embedding_size = 300
    self.pos_size = 56
    self.pos_dim = 12
    self.ner_size = 19
    self.ner_dim = 8
    self.vocab_size = 25000

    self.word_base = False
    self.dropout = 0.6
    self.rnn_layer = 1
    self.batch_size = 32
    self.lr = 5e-3
    self.valid_ratio = 0.05
    self.valid_iters = 1
    self.max_sent = 25
    self.display_freq = 10
    self.save_freq = 1
    self.epoch = 24
    self.init_embedding = False
    self.embedding_source = './'


use_cuda = torch.cuda.is_available()
# parser = argparse.ArgumentParser(description='Model and Training parameters')
# # Model Architecture
# parser.add_argument('--rnn_type', type=str, default='lstm', help='the rnn cell used')
# parser.add_argument('--hidden_size', type=int, default=256, help='the hidden size of RNNs [256]')
# parser.add_argument('--embedding_size', type=int, default=200, help='the embedding size [200]')
# parser.add_argument('--vocab_size', type=int, default=25000, help='the vocab size [25000]')
# # Training hyperparameter
# parser.add_argument('--word_base', action='store_true')
# parser.add_argument('--dropout', type=float, default=0.6)
# parser.add_argument('--rnn_layer', type=int, default=1)
# parser.add_argument('--batch_size', type=int, default=32, help='the size of batch [32]')
# parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate of encoder [1e-3]')
# parser.add_argument('--valid_ratio', type=float, default=0.05)
# parser.add_argument('--valid_iters', type=int, default=1, help='run validation batch every N epochs [1]')
# parser.add_argument('--max_sent', type=int, default=25, help='max length of encoder, decoder')
# parser.add_argument('--display_freq', type=int, default=10, help='display training status every N iters [10]')
# parser.add_argument('--save_freq', type=int, default=1, help='save model every N epochs [1]')
# parser.add_argument('--epoch', type=int, default=25, help='train for N epochs [25]')
# parser.add_argument('--init_embedding', action='store_true', help='whether init embedding')
# parser.add_argument('--embedding_source', type=str, default='./', help='pretrained embedding path')


args = Arg()

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    train = load_data('/Users/lap60754/Documents/UIT_DSC2023/ise-dsc01/data_preprocessed/train_claim_ner_pos.json')
    test = load_data('/Users/lap60754/Documents/UIT_DSC2023/ise-dsc01/data_preprocessed/public_test_claim_ner_pos.json')
    vocabulary, pad_lens = build_vocab(train, test, args.vocab_size)
    # embedding = vocabulary.build_embedding('/kaggle/input/phoword2vec-vi-words/word2vec_vi_words_300dims.txt', args.embedding_size)
    # print(embedding.shape)
    print('Vocab size: %d | Max context: %d | Max claim: %d'%(
          len(vocabulary), pad_lens[0], pad_lens[1]))
    # valid, test = split_exp(test, 0.5)
    print('Train: %d | Test: %d'%(len(train['ids']), len(test['ids'])))
    train_engine = DataLoader(DataEngine(train, vocabulary, pad_lens),
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=use_cuda)
    test_engine = DataEngine(test, vocabulary, pad_lens)
    
    if args.init_embedding:
        w2v = load_embedding(args.embedding_source, 
                       vocabulary.to_idx,
                       args.embedding_size)
    else:
        w2v = None
    # fusion_net = FusionNet(vocab_size=len(vocabulary),
    #                        pos_size=vocabulary.pos_size(),
    #                        pos_dim=args.pos_dim,
    #                        ner_size=vocabulary.ner_size(),
    #                        ner_dim=args.ner_dim,
    #                        word_dim=args.embedding_size,
    #                        hidden_size=args.hidden_size,
    #                        rnn_layer=args.rnn_layer,
    #                        dropout=args.dropout,
    #                        pretrained_embedding=embedding)

    print("created netword...")
    