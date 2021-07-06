#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx, jby
@Date: 2020-07-13 11:07:48
@LastEditTime: 2020-07-18 14:25:15
@LastEditors: Please set LastEditors
@Description: Helper functions or classes used for the model.
@FilePath: /JD_project_2/baseline/model/utils.py
'''

import numpy as np
import time
import heapq
import random
import sys
import pathlib

import torch

import matplotlib.pyplot as plt

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

import config

# import matplotlib
# matplotlib.use('Agg')

def abstract2ids(abstract_words, vocab, source_oovs):
    """Map tokens in the abstract (reference) to ids.
       OOV tokens in the source will be remained.

    Args:
        abstract_words (list): Tokens in the reference.
        vocab (vocab.Vocab): The vocabulary.
        source_oovs (list): OOV tokens in the source.

    Returns:
        list: The reference with tokens mapped into ids.
    """
    output_ids = []
    unk_id = vocab.UNK
    
    for word in abstract_words:
        word_id = vocab[word]
        if word_id != unk_id:
            output_ids.append(word_id)
        else:  # UNK in vocab
            if word in source_oovs:
                output_ids.append(source_oovs.index(word) + vocab.size())
            else:  # if word is not even in the source
                output_ids.append(unk_id)
                
    return output_ids
            
#     ids = []
#     oovs = []
#     unk_id = vocab["<UNK>"]
    
#     for word in source_words:
#         word_id = vocab[word]
#         if word_id == unk_id:  # if word is oov
#             if word not in oovs:
#                 oovs.append(word)
#             oov_num = oovs.index(word)
#             ids.append(vocab.size() + oov_num)
#         else:
#             ids.append(word_id)
            
#     return ids, oovs
    

def timer(module):
    """Decorator function for a timer.

    Args:
        module (str): Description of the function being timed.
    """
    def wrapper(func):
        """Wrapper of the timer function.

        Args:
            func (function): The function to be timed.
        """
        def cal_time(*args, **kwargs):
            """The timer function.

            Returns:
                res (any): The returned value of the function being timed.
            """
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time
    return wrapper


def simple_tokenizer(text):
    return text.split()


def count_words(counter, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence:
            counter[word] += 1


def sort_batch_by_len(data_batch):
    """

    Args:
        data_batch (Tensor): Batch before sorted.

    Returns:
        Tensor: Batch after sorted.
    """
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # Sort indices of data in batch by lengths.
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    data_batch = {
        name: [_tensor[i] for i in sorted_indices]
        for name, _tensor in res.items()
    }
    return data_batch


def outputids2words(id_list, source_oovs, vocab):
    """
        Maps output ids to words, including mapping in-source OOVs from
        their temporary ids to the original OOV string (applicable in
        pointer-generator mode).
        Args:
            id_list: list of ids (integers)
            vocab: Vocabulary object
            source_oovs:
                list of OOV words (strings) in the order corresponding to
                their temporary source OOV ids (that have been assigned in
                pointer-generator mode), or None (in baseline mode)
        Returns:
            words: list of words (strings)
    """

    ###########################################
    #          TODO: module 1 task 4          #
    ###########################################

    words = []
    
    for id in id_list:
        try:
            word = vocab.index2word[id]
        except IndexError:
            assert_msg = "Error: cannot find the ID the in the vocabulary."
            assert source_oovs is not None, assert_msg
            source_oov_idx = id - vocab.size()
            try:
                word = source_oovs[source_oov_idx]
            except ValueError:
                raise ValueError('Error: model produced word ID %i corresponding to source OOV %i \
                     but this example only has %i source OOVs'
                    % (i, source_oov_idx, len(source_oovs)))
        
        words.append(word)
        
        
#         if id >= vocab.size():  # oov temporary id
#             words.append(source_oovs[id - vocab.size()])
#         else:
#             words.append(vocab[id])

    return ' '.join(words)


def source2ids(source_words, vocab):
    """Map the source words to their ids and return a list of OOVs in the source.
    Args:
        source_words: list of words (strings)
        vocab: Vocabulary object
    Returns:
    ids:
        A list of word ids (integers); OOVs are represented by their temporary
        source OOV number. If the vocabulary size is 50k and the source has 3
        OOVs tokens, then these temporary OOV numbers will be 50000, 50001,
        50002.
    oovs:
        A list of the OOV words in the source (strings), in the order
        corresponding to their temporary source OOV numbers.
    """

    ###########################################
    #          TODO: module 1 task 3          #
    ###########################################

    ids = []
    oovs = []
    unk_id = vocab["<UNK>"]
    
    for word in source_words:
        word_id = vocab[word]
        if word_id == unk_id:  # if word is oov
            if word not in oovs:
                oovs.append(word)
            oov_num = oovs.index(word)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(word_id)
            
    return ids, oovs


class Beam(object):
    """The contianer for a temperay sequence used in beam search.
    """
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 coverage_vector):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    def extend(self,
               token,
               log_prob,
               decoder_states,
               coverage_vector):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    def seq_score(self):
        """
        This function calculate the score of the current sequence.
        The scores are calculated according to the definitions in
        https://opennmt.net/OpenNMT/translation/beam_search/.
        1. Lenth normalization is used to normalize the cumulative score
        of a whole sequence.
        2. Coverage normalization is used to favor the sequences that fully
        cover the information in the source. (In this case, it serves different
        purpose from the coverage mechanism defined in PGN.)
        3. Alpha and beta are hyperparameters that used to control the
        strengths of ln and cn.
        """
        Y_len = len(self.tokens)
        # Length Normalization
        lp = (5 + Y_len)**config.alpha / (5 + 1)**config.alpha
        
        # Coverage Normalization
        cp = config.beta * torch.sum(
            torch.log(config.eps + 
                torch.where(self.coverage_vector < 1.0, self.coverage_vector, torch.ones((1, self.coverage_vector.shape[1])).to(torch.device(config.DEVICE)))
            )
        )

        score = sum(self.log_probs) / lp + cp
        return score

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()


def add2heap(heap, item, k):
    """Maintain a heap with k nodes and the smallest one as root.

    Args:
        heap (list): The list to heapify.
        item (tuple):
            The tuple as item to store.
            Comparsion will be made according to values in the first position.
            If there is a tie, values in the second position will be compared,
            and so on.
        k (int): The capacity of the heap.
    """
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)
        

def replace_oovs(in_tensor, vocab):
    """Replace oov tokens in a tensor with the <UNK> token.

    Args:
        in_tensor (Tensor): The tensor before replacement.
        vocab (vocab.Vocab): The vocabulary.

    Returns:
        Tensor: The tensor after replacement.
    """    
    oov_token = torch.full(in_tensor.shape, vocab.UNK).long().to(config.DEVICE)
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor


def plot_grad_flow(named_parameters):
#     matplotlib.use('Agg')
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(config.gradflow_save_name + 'grad_flow.png')
    
    
class ScheduledSampler():
    def __init__(self, phases):
        self.phases = phases
        self.scheduled_probs = [i / (self.phases - 1) for i in range(self.phases)]

    def teacher_forcing(self, phase):
        """According to a certain probability to choose whether to execute teacher_forcing

        Args:
            phase (int): probability level  if phase = 0, 100% teacher_forcing ,phase = self.phases - 1, 0% teacher_forcing 

        Returns:
            bool: teacher_forcing or not 
        """
        ###########################################
        #          TODO: module 5 task 1          #
        ###########################################
        rand_probs = random.random()
        if rand_probs >= self.scheduled_probs[phase]:
            return True
        else:
            return False
        
def config_info(config):
    """get some config information

    Args:
        config (model): define in  model/config.py
    Returns:
        string: config information
    """
    info = 'model_name = {}, pointer = {}, coverage = {}, fine_tune = {}, scheduled_sampling = {}, weight_tying = {},' +\
          'source = {}  '
    return (info.format(config.model_name, config.pointer, config.coverage, config.fine_tune, config.scheduled_sampling,
                      config.weight_tying, config.source))