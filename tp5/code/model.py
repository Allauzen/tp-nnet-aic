"""
Minimal gram-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License

Modified by Gaetan Marceau Caron [02/20/2016]
            
Usage: rnn [-m <model>] [-f <file>] [-o <order>] [-e <eta>] [-s <length>] [-n <nhidden>]

Options:
-h --help      Show the description of the program
-f <file> --filename <file>  filename of the text to learn [default: Dostoevsky.txt]
-o <order> --order <order>  order of the model[default: 1]
-e <eta> --eta <eta>  step-size for the optimizer[default: 1e-1]
-s <length> --seq_length <length>  sequence length[default: 25]
-n <nhidden> --nhidden <nhidden>  number of hidden units[default: 100] 
"""
import argparse, re
import numpy as np
import math
from collections import Counter 

from docopt import docopt

def buildGramAlphabet(text, order=1):
  gram_dic = Counter()
  for i in range(0, len(text)):
    gram = text[i:i+order]
    gram_dic[gram] = gram_dic.get(gram, 0) + 1
  return gram_dic

def preprocess(text):
  text = re.sub("\s\s+", " ", text)
  text = re.sub("\n", " ", text)
  text = re.sub("_", "", text)
  text = re.sub("à", "a", text)
  text = re.sub("ô", "o", text)
  text = re.sub("é", "e", text)
  text = re.sub("è", "e", text)
  text = re.sub("ë", "e", text)
  text = re.sub("ê", "e", text)
  text = re.sub("î", "i", text)
  text = re.sub("ï", "i", text)
  text = re.sub("\*", "", text)
  text = re.sub("!", ".", text)
  text = re.sub("ç", "c", text)
  text = re.sub("-", " ", text)
  text = re.sub("\[", " ", text)
  text = re.sub("\]", " ", text)
  text = re.sub("\(", " ", text)
  text = re.sub("\)", " ", text)
  text = re.sub("\?", ".", text)
  text = re.sub("'", " ", text)
  text = re.sub(";", " ", text)
  text = re.sub("ü", "u", text)
  return text.lower()

# gradient checking
from random import uniform
def gradCheck(inputs, target, hprev):
  global Wxh, Whh, Why, bh, by
  num_checks, delta = 10, 1e-5
  _, dWxh, dWhh, dWhy, dbh, dby, _ = lossFun(inputs, targets, hprev)
  for param,dparam,name in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], ['Wxh', 'Whh', 'Why', 'bh', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    #assert(s0 == s1, 'Error dims dont match: %s and %s.' % (s0, s1))
    print(name)
    for i in range(num_checks):
      ri = int(uniform(0,param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val - delta
      cg1, _, _, _, _, _, _ = lossFun(inputs, targets, hprev)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      print('%f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
      if grad_numerical > 0.:
        assert(rel_error < 1e-3)
      # rel_error should be on order of 1e-7 or less

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0

  # forward pass
  for t in range(len(inputs)):
    ############ complete with eq.2 ############
    ############ complete with eq.3 ############
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    ############ complete ############
    ############ complete ############
    ############ complete ############
    ############ complete ############
    ############ complete ############
    temp = np.zeros_like(dWxh)
    temp[:,inputs[t]] = dhraw.ravel()
    dWxh += temp
    ############ complete ############
    ############ complete ############
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((alphabet_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(alphabet_size), p=p.ravel())
    x = np.zeros((alphabet_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

if __name__ == '__main__':

    # Retrieve the arguments from the command-line
    args = docopt(__doc__)
    print(args)

    # data I/O
    data = preprocess(open(args["--filename"], 'r').read()) # should be simple plain text file
    model_order = int(args["--order"])
    gram_dic = buildGramAlphabet(data,model_order)
    print(gram_dic)

    data_size, alphabet_size = len(data), len(gram_dic)
    print('data has %d characters and alphabet has %d symbols' % (data_size, alphabet_size))
    char_to_ix = { ch:i for i,ch in enumerate(gram_dic.keys()) }
    ix_to_char = { i:ch for i,ch in enumerate(gram_dic.keys()) }
    
    # hyperparameters
    hidden_size = int(args["--nhidden"]) # size of hidden layer of neurons
    seq_length = int(args["--seq_length"]) # number of steps to unroll the RNN for
    eta = float(args["--eta"])
    
    # initialization of the parameters
    Wxh = np.random.randn(hidden_size, alphabet_size)*0.01 # input to hidden
    Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
    Why = np.random.randn(alphabet_size, hidden_size)*0.01 # hidden to output
    bh = np.zeros((hidden_size, 1)) # hidden bias
    by = np.zeros((alphabet_size, 1)) # output bias
    
    n, p = 0, 0
    mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
    smooth_loss = -np.log(1.0/alphabet_size)*seq_length # loss at iteration 0
    while True:
      # prepare inputs (we're sweeping from left to right in steps seq_length long)
      if p+seq_length*model_order+1 >= len(data) or n == 0: 
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        p = 0 # go from start of data
      inputs = [char_to_ix[ch] for ch in [data[p+(i*model_order):p+((i+1)*model_order)] for i in range(seq_length)]]
      targets = [char_to_ix[ch] for ch in [data[p+((i+1)*model_order):p+((i+2)*model_order)] for i in range(seq_length)]]

      gradCheck(inputs, targets, hprev)       ### comment only when gradCheck is ok!
      
      # sample from the model now and then
      if n % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

      # forward seq_length characters through the net and fetch gradient
      loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if n % 100 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                    [dWxh, dWhh, dWhy, dbh, dby], 
                                    [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -eta * dparam / np.sqrt(mem + 1e-8) # adagrad update

      p += seq_length # move data pointer
      n += 1 # iteration counter 
