import argparse
import os
import time
import math
import collections
import pickle as pkl
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LogisticRegression

from fp16 import FP16_Module, FP16_Optimizer
from apex.reparameterization import apply_weight_norm, remove_weight_norm

from model import DistributedDataParallel as DDP
from model import RNNFeaturizerHist as RNNFeaturizerHist
from model import LSTMClassifier as LSTMClassifier
from model import LRClassifier as LRClassifier

from configure_data import configure_data

parser = argparse.ArgumentParser(description='PyTorch Sentiment Discovery Transfer Learning')

parser.add_argument('--exp', type=str, default='LR',
                    help='experiment to run (LR, CNN')
parser.add_argument('--model', type=str, default='mLSTM',
                    help='type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM, GRU')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=4096,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--all_layers', action='store_true',
                    help='if more than one layer is used, extract features from all layers, not just the last layer')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to run Logistic Regression')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--load_model', type=str,  default='lang_model.pt', required=True,
                    help='path to trained world language model')
parser.add_argument('--save_results', type=str,  default='sentiment',
                    help='path to save intermediate and final results of transfer')
parser.add_argument('--fp16', action='store_true',
                    help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--neurons', default=1, type=int,
                    help='number of nenurons to extract as features')
parser.add_argument('--no_test_eval', action='store_true',
                    help='whether to not evaluate the test model (useful when your test set has no labels)')
parser.add_argument('--write_results', default='',
                    help='write results of model on test (or train if none is specified) data to specified filepath [only supported for csv datasets currently]')
parser.add_argument('--use_cached', action='store_true',
                    help='reuse cached featurizations from a previous from last time')
parser.add_argument('--drop_neurons', action='store_true',
                    help='drop top neurons instead of keeping them')
parser.add_argument('--cuda', action='store_true',
                    help='use cuda acceleration')
data_config, data_parser = configure_data(parser)

args = parser.parse_args()

exp_dataset = args.exp.split('_')[0]
exp_model = args.exp.split('_')[1]
exp_name = args.exp #+ '_' + args.data.split('/')[1]
print('Running:', exp_name)

if exp_dataset == 'sst':
    data_parser.set_defaults(split='1.', data='data/binary_sst/train.csv')
    data_parser.set_defaults(valid='data/binary_sst/val.csv', test='data/binary_sst/test.csv')
elif exp_dataset == 'imdb':
    data_parser.set_defaults(split='1.', data='data/imdb/train.json')
    data_parser.set_defaults(valid='data/imdb/test.json', test='data/imdb/test.json')    

args = parser.parse_args()


args.cuda = torch.cuda.is_available()
if not args.cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
if args.seed is not -1:
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

train_data, val_data, test_data = data_config.apply(args)
ntokens = args.data_size
model = RNNFeaturizerHist(args.model, ntokens, args.emsize, args.nhid, args.nlayers, 0.0, args.all_layers)
if args.cuda:
    model.cuda()

if args.fp16:
    model.half()

# load char embedding and recurrent encoder for featurization
with open(args.load_model, 'rb') as f:
    sd = x = torch.load(f)
    if 'encoder' in sd:
        sd = sd['encoder']

try:
    model.load_state_dict(sd)
except:
    # if state dict has weight normalized parameters apply and remove weight norm to model while loading sd
    apply_weight_norm(model.rnn)
    model.load_state_dict(sd)
    remove_weight_norm(model)

def get_batch_transform(model, text):
    '''
    Apply featurization `model` to extract features from text in data loader.
    Featurization model should return cell state not hidden state.
    `text` data loader should return tuples of ((text, text length), text label)
    Returns labels and features for samples in text.
    '''
    model.eval()

    def get_batch(batch):
        '''
        Process batch and return tuple of (text, text label, text length) long tensors.
        Text is returned in column format with (time, batch) dimensions.
        '''
        (text, timesteps), labels = batch
        text = Variable(text).long()
        timesteps = Variable(timesteps).long()
        labels = Variable(labels).long()
        if args.cuda:
            text, timesteps, labels = text.cuda(), timesteps.cuda(), labels.cuda()
        return text.t(), labels.float(), timesteps-1

    n = 0
    len_ds = len(text)
    # Use no grad context for improving memory footprint/speed of inference
    #with torch.no_grad():
    if True:    
        for i, data in enumerate(text):
            torch.set_grad_enabled(False)
            text_batch, labels_batch, length_batch = get_batch(data)
            # get batch size and reset hidden state with appropriate batch size
            batch_size = text_batch.size(1)
            n += batch_size
            model.rnn.reset_hidden(batch_size)
            # extract batch of features from text batch
            cell, hist = model(text_batch, length_batch)
            cell = cell.float()
            torch.set_grad_enabled(True)
            yield length_batch, hist, labels_batch


save_root = args.load_model
save_root = save_root.replace('.current', '')
save_root = os.path.splitext(save_root)[0]
save_root += '_transfer'
save_root = os.path.join(save_root, args.save_results)
if not os.path.exists(save_root):
    os.makedirs(save_root)
#print('writing results to '+save_root)



# create data generators
generator_train = get_batch_transform(model, train_data)
generator_val = get_batch_transform(model, val_data)
generator_test = get_batch_transform(model, test_data)

# train a predictor model
if exp_model == 'LR':
    model_snt = LRClassifier()
elif exp_model == 'LSTM':
    model_snt = LSTMClassifier()
else:
    raise NotImplementedError
if args.cuda:
    model_snt.cuda()

iters_max = 4000
lr_base = 0.001
lr_final = 0.00005
lr_new = lr_base
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model_snt.parameters(), lr=lr_base)

iters_val = []
accus_val = []
loss_val = []

for iter_cnt in range(iters_max+1):
    # get a batch
    try:
        trL, trX, trY = generator_train.__next__()
    except StopIteration:
        generator_train = get_batch_transform(model, train_data)
        trL, trX, trY = generator_train.__next__()

    optimizer.zero_grad()
    output = model_snt(trX)
    # backward path
    loss = criterion(output, trY)
    #print('Loss', loss)
    loss.backward()#retain_graph=True)
    #pdb.set_trace()
    optimizer.step()
    print('Training iter # {}, lr: {:e}, loss: {:f}'.format(iter_cnt, lr_new, loss), end='\r')
    del loss
    del output
    del trL, trX, trY
    # if validation iter
    if iter_cnt % 50 == 0:
        torch.cuda.empty_cache()
        try:
            vaL, vaX, vaY = generator_val.__next__()
        except StopIteration:
            generator_val = get_batch_transform(model, val_data)
            vaL, vaX, vaY = generator_val.__next__()
        #pdb.set_trace()
        output = model_snt(vaX)
        pred = (output > 0.5).float()
        loss = criterion(output, vaY)
        accu = torch.mean((pred==vaY).float())
        iters_val.append(iter_cnt)
        accus_val.append(100.0*accu.item())
        loss_val.append(loss.item())
        print('\n Validation Accuracy: {}'.format(accu))
        del accu
        del loss
        del output
        del vaL, vaX, vaY
        #pdb.set_trace()
    if iter_cnt % 250 == 0:
        lr_new = (iter_cnt/iters_max)*(lr_final-lr_base) + lr_base
        optimizer = torch.optim.Adam(model_snt.parameters(), lr=lr_new)
        # save the current results
        np.savetxt('./run_outputs/sentiment_'+exp_name+'.csv', 
                       np.vstack([iters_val, loss_val, accus_val]).T, 
                       delimiter=',', header='iters_val, loss_val, accus_val')
pdb.set_trace()
