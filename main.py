import argparse
import os
import sys
import time
import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim

from fp16 import FP16_Module, FP16_Optimizer

import data
import model
from model import DistributedDataParallel as DDP
import data_utils.preprocess

from apex.reparameterization import apply_weight_norm, remove_weight_norm
from configure_data import configure_data
from learning_rates import LinearLR

parser = argparse.ArgumentParser(description='PyTorch Sentiment-Discovery Language Modeling')
parser.add_argument('--exp', type=str, default='original',
                    help='the experiment to run: (original,pretrain)')
parser.add_argument('--model', type=str, default='mLSTM',
                    help='type of recurrent net (RNNTanh, RNNReLU, LSTM, mLSTM, GRU)')
parser.add_argument('--emsize', type=int, default=64,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=4096,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='lang_model.pt',
                    help='path to save the final model')
parser.add_argument('--load', type=str, default='',
                    help='path to a previously saved model checkpoint')
parser.add_argument('--save_iters', type=int, default=2000, metavar='N',
                    help='save current model progress interval')
parser.add_argument('--fp16', action='store_true',
                    help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--dynamic_loss_scale', action='store_true',
                    help='Dynamically look for loss scalar for fp16 convergance help.')
parser.add_argument('--no_weight_norm', action='store_true',
                    help='Add weight normalization to model.')
parser.add_argument('--loss_scale', type=float, default=1,
                    help='Static loss scaling, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--world_size', type=int, default=1,
                    help='number of distributed workers')
parser.add_argument('--distributed_backend', default='gloo',
                    help='which backend to use for distributed training. One of [gloo, nccl]')
parser.add_argument('--rank', type=int, default=-1,
                    help='distributed worker rank. Typically set automatically from multiproc.py')
parser.add_argument('--optim', default='Adam',
                    help='One of PyTorch\'s optimizers (Adam, SGD, etc). Default: Adam')

# Add dataset args to argparser and set some defaults
data_config, data_parser = configure_data(parser)
data_config.set_defaults(data_set_type='unsupervised', transpose=True)
data_parser.set_defaults(split='100,1,1')
data_parser = parser.add_argument_group('language modeling data options')
data_parser.add_argument('--seq_length', type=int, default=256,
                         help="Maximum sequence length to process (for unsupervised rec)")
data_parser.add_argument('--eval_seq_length', type=int, default=256,
                         help="Maximum sequence length to process for evaluation")
data_parser.add_argument('--lazy', action='store_true',
                         help='whether to lazy evaluate the data set')
data_parser.add_argument('--persist_state', type=int, default=1,
                         help='0=reset state after every sample in a shard, 1=reset state after every shard, -1=never reset state')
data_parser.add_argument('--num_shards', type=int, default=102,
                         help="""number of total shards for unsupervised training dataset. If a `split` is specified,
                                 appropriately portions the number of shards amongst the splits.""")
data_parser.add_argument('--val_shards', type=int, default=0,
                         help="""number of shards for validation dataset if validation set is specified and not split from training""")
data_parser.add_argument('--test_shards', type=int, default=0,
                         help="""number of shards for test dataset if test set is specified and not split from training""")


args = parser.parse_args()

torch.backends.cudnn.enabled = False
args.cuda = torch.cuda.is_available()

# initialize distributed process group and set device
if args.rank > 0:
    torch.cuda.set_device(args.rank % torch.cuda.device_count())

if args.world_size > 1:
    distributed_init_file = os.path.splitext(args.save)[0]+'.distributed.dpt'
    torch.distributed.init_process_group(backend=args.distributed_backend, world_size=args.world_size,
                                                    init_method='file://'+distributed_init_file, rank=args.rank)

# Set the random seed manually for reproducibility.
if args.seed is not -1:
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

if args.loss_scale != 1 and args.dynamic_loss_scale:
    raise RuntimeError("Static loss scale and dynamic loss scale cannot be used together.")
    

###############################################################################
# Pre-Training code
###############################################################################
if args.exp == 'pretrain':
    print('Pretraining...')
    data_pretrain = {'word':[], 'vec':[], 'seq':[]}
    with open('/home/mohammad/Database/NLP/glove.6B/glove.6B.300d.txt', 'r') as f:
        dfile = f.readlines()
    for line in dfile[:]:
        line = line.strip('\n').split(' ', 1)
        word = line[0]
        vec = np.fromstring(line[1], sep=' ')
        seq = data_utils.preprocess.tokenize_str_batch([word])
        try:
            seq = seq[0].narrow(1, 2, seq[1]-3)
        except:
            continue
        data_pretrain['word'].append(word)
        data_pretrain['vec'].append(vec)
        data_pretrain['seq'].append(seq)


    def get_batch_pretrain(data, size=128):
        inds = np.random.choice(len(data['word']), size=size)
        features = []
        targets = []
        for ind in inds:
            if args.cuda:
                feature = data['seq'][ind].long().cuda()
                target = torch.tensor(data['vec'][ind], dtype=torch.float).cuda()
            else:
                feature = data['seq'][ind].long()
                target = torch.tensor(data['vec'][ind], dtype=torch.float)         
            features.append(feature)
            targets.append(target)
        return features, targets

    features_pretrn, targets_pretrn = get_batch_pretrain(data_pretrain, size=128)

    criterion_pre = nn.MSELoss()

    ntokens = args.data_size
    model_pre = model.RNNModelPreTrain(args.model, ntokens, args.emsize, 
                                       args.nhid, args.nlayers, args.dropout, args.tied, nvec=300)
    if args.cuda:
        model_pre.cuda()

    # create optimizer and fp16 models
    if args.fp16:
        model_pre = FP16_Module(model_pre)
        optim_pre = eval('torch.optim.'+args.optim)(model_pre.parameters(), lr=0.0001)
        optim_pre = FP16_Optimizer(optim_pre, 
                               static_loss_scale=args.loss_scale,
                               dynamic_loss_scale=args.dynamic_loss_scale)
    else:
        optim_pre = eval('torch.optim.'+args.optim)(model_pre.parameters(), lr=0.0001)

    # # add linear learning rate scheduler
    # if train_data is not None:
    #     num_iters = len(train_data) * args.epochs
    #     LR = LinearLR(optim, num_iters)   

    # wrap model for distributed training
    if args.world_size > 1:
        model_pre = DDP(model_pre)


    for iter_trn in range(10000):
        # get a train batch
        features_pretrn, targets_pretrn = get_batch_pretrain(data_pretrain, size=128)
        total_loss = 0.0
        # zero grads
        model_pre.zero_grad()
        optim_pre.zero_grad()
        for feature_seq, target_vec in zip(features_pretrn, targets_pretrn):
            # clear hidden state
            model_pre.hidden = model_pre.init_hidden()
            # do forward path
            pred_vec = model_pre(feature_seq)[0][-1,-1]
            loss = criterion_pre(pred_vec, target_vec)
            total_loss += loss.data.float()
            # do backward path
            # loss.backward()
            # optimize
            if args.fp16:
                optim_pre.backward(loss)
            else:
                loss.backward(retain_graph=True)
        # clipping gradients helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip > 0:
            if not args.fp16:
                torch.nn.utils.clip_grad_norm(model_pre.parameters(), args.clip)
            else:
                optim_pre.clip_fp32_grads(clip=args.clip)
        optim_pre.step()
        print('Iter {:3d}, loss {:.2E}'.format(iter_trn, total_loss))
    #pdb.set_trace()

###############################################################################
# Load data
###############################################################################

# Starting from sequential data, the unsupervised dataset type loads the corpus
# into rows. With the alphabet as the our corpus and batch size 4, we get
# ┌ a b c d e f ┐
# │ g h i j k l │
# │ m n o p q r │
# └ s t u v w x ┘.
# These rows are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
#
# The unsupervised dataset further splits the corpus into shards through which
# the hidden state is persisted. The dataset also produces a hidden state 
# reset mask that resets the hidden state at the start of every shard. A valid
# mask might look like
# ┌ 1 0 0 0 0 0 ... 0 0 0 1 0 0 ... ┐
# │ 1 0 0 0 0 0 ... 0 1 0 0 0 0 ... │
# │ 1 0 0 0 0 0 ... 0 0 1 0 0 0 ... │
# └ 1 0 0 0 0 0 ... 1 0 0 0 0 0 ... ┘.
# With 1 indicating to reset hidden state at that particular minibatch index
 
train_data, val_data, test_data = data_config.apply(args)
###############################################################################
# Build the model
###############################################################################

ntokens = args.data_size
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
print('* number of parameters: %d' % sum([p.nelement() for p in model.parameters()]))
if args.cuda:
    model.cuda()

rnn_model = model

if args.exp == 'pretrain':
    sd = model_pre.state_dict()
    model.load_state_dict(sd)
    # free-up gpu memory
    del model_pre
    torch.cuda.empty_cache()
else:
    if args.load != '':
        sd = torch.load(args.load)
        try:
            model.load_state_dict(sd)
        except:
            apply_weight_norm(model.rnn, hook_child=False)
            model.load_state_dict(sd)
            remove_weight_norm(model.rnn)

if not args.no_weight_norm:
    apply_weight_norm(model.rnn, hook_child=False)

# create optimizer and fp16 models
if args.fp16:
    model = FP16_Module(model)
    optim = eval('torch.optim.'+args.optim)(model.parameters(), lr=args.lr)
    optim = FP16_Optimizer(optim, 
                           static_loss_scale=args.loss_scale,
                           dynamic_loss_scale=args.dynamic_loss_scale)
else:
    optim = eval('torch.optim.'+args.optim)(model.parameters(), lr=args.lr)

# add linear learning rate scheduler
if train_data is not None:
    num_iters = len(train_data) * args.epochs
    LR = LinearLR(optim, num_iters)

# wrap model for distributed training
if args.world_size > 1:
    model = DDP(model)

criterion = nn.CrossEntropyLoss()



###############################################################################
# Training code
###############################################################################

# get_batch subdivides the source data into chunks of length args.seq_length.
# If source is equal to the example output of the data loading example, with
# a seq_length limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the data loader. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM. A Variable representing an appropriate 
# shard reset mask of the same dimensions is also returned.


def get_batch(data):
    reset_mask_batch = data[1].long()
    data = data[0].long()
    if args.cuda:
        data = data.cuda()
        reset_mask_batch = reset_mask_batch.cuda()
    text_batch = Variable(data[:,:-1].t().contiguous(), requires_grad=False)
    target_batch = Variable(data[:,1:].t().contiguous(), requires_grad=False)
    reset_mask_batch = Variable(reset_mask_batch[:,:text_batch.size(0)].t().contiguous(), requires_grad=False)
    return text_batch, target_batch, reset_mask_batch

def init_hidden(batch_size):
    return rnn_model.rnn.init_hidden(args.batch_size)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    init_hidden(args.batch_size)
    total_loss = 0
    ntokens = args.data_size
    with torch.no_grad():
        for i, batch in enumerate(data_source):
            data, targets, reset_mask = get_batch(batch)
            output, hidden = model(data, reset_mask=reset_mask)
            output_flat = output.view(-1, ntokens).contiguous().float()
            total_loss += criterion(output_flat, targets.view(-1).contiguous()).data[0]
    return total_loss / max(len(data_source), 1)

iters_count = 0

def train(total_iters=0):
    global iters_count
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = args.data_size
    hidden = init_hidden(args.batch_size)
    curr_loss = 0.
    for i, batch in enumerate(train_data):
        iters_count += 1
        data, targets, reset_mask = get_batch(batch)
        output, hidden = model(data, reset_mask=reset_mask)
        loss = criterion(output.view(-1, ntokens).contiguous().float(), targets.view(-1).contiguous())

        optim.zero_grad()

        if args.fp16:
            optim.backward(loss)
        else:
            loss.backward()
        total_loss += loss.data.float()


        # clipping gradients helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip > 0:
            if not args.fp16:
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            else:
                optim.clip_fp32_grads(clip=args.clip)
        optim.step()

        # step learning rate and log training progress
        lr = LR.get_lr()[0]
        if not args.fp16:
            LR.step()
        else:
            # if fp16 optimizer skips gradient step due to explosion do not step lr
            if not optim.overflow:
                LR.step()

        if i % args.log_interval == 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2E} | ms/batch {:.3E} | \
                  loss {:.2E} | ppl {:8.2f} | loss scale {:8.2f}'.format(
                      epoch, i, len(train_data), lr,
                      elapsed * 1000 / args.log_interval, cur_loss, math.exp(min(cur_loss, 20)),
                      args.loss_scale if not args.fp16 else optim.loss_scale 
                  )
            )
            # write logs to a file
            with open('./run_outputs/'+args.exp+'.txt', 'a+') as f:
                f.write(str(epoch)+', '+str(iters_count)+', '+str(cur_loss.item())+'\n')
            total_loss = 0
            start_time = time.time()
            sys.stdout.flush()

        # save current model progress. If distributed only save from worker 0
        if args.save_iters and total_iters % (args.save_iters) == 0 and total_iters > 0 and args.rank < 1:
            if args.rank < 1:
                with open(os.path.join(os.path.splitext(args.save)[0], 'e%s.pt'%(str(total_iters),)), 'wb') as f:
                    torch.save(model.state_dict(), f)
            if args.cuda:
                torch.cuda.synchronize()
        total_iters += 1

    return cur_loss

# Loop over epochs.
lr = args.lr
best_val_loss = None

# If saving process intermittently create directory for saving
if args.save_iters > 0 and not os.path.exists(os.path.splitext(args.save)[0]) and args.rank < 1:
    os.makedirs(os.path.splitext(args.save)[0])

# At any point you can hit Ctrl + C to break out of training early.
try:
    total_iters = 0
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        val_loss = train(total_iters)
        total_iters += len(train_data)
        if val_data is not None:
            print('entering eval')
            val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(min(val_loss, 20))))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss and args.rank <= 0:
            with open(args.save, 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val_loss = val_loss
        torch.cuda.synchronize()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
if os.path.exists(args.save):
    with open(args.save, 'rb') as f:
        model.load_state_dict(torch.load(f))

if not args.no_weight_norm and args.rank <= 0:
    remove_weight_norm(rnn_model)
    with open(args.save, 'wb') as f:
        torch.save(model.state_dict(), f)
if args.cuda:
    torch.cuda.synchronize()

if test_data is not None:
    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
