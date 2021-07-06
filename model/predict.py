#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx, jby
@Date: 2020-07-13 11:00:51
@LastEditTime: 2020-07-18 16:48:37
@LastEditors: Please set LastEditors
@Description: Generate a summary.
@FilePath: /JD_project_2/baseline/model/predict.py
'''


import random

import torch
import jieba
import pathlib
import sys
import heapq


abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

import config
from model import Seq2seq, PGN
from dataset import PairDataset
from utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs


class Predict():
    @timer(module='initalize predicter')
    def __init__(self):
        self.DEVICE = torch.device("cuda" if config.is_cuda else "cpu")
        dataset = PairDataset(config.data_path,
                              max_src_len=config.max_src_len,
                              max_tgt_len=config.max_tgt_len,
                              truncate_src=config.truncate_src,
                              truncate_tgt=config.truncate_tgt)

        self.vocab = dataset.build_vocab(embed_file=config.embed_file)

        if 'baseline' in config.model_name:
            self.model = Seq2seq(self.vocab)
        else:
            self.model = PGN(self.vocab)
            
        self.stop_word = list(
            set([
                self.vocab[x.strip()] for x in
                open(config.stop_word_file
                     ).readlines()
            ]))
        
        print('loading ' + config.model_name + '.....')
        self.model.load_model()
        self.model.to(self.DEVICE)

    def greedy_search(self,
                      encoder_input,
                      max_sum_len,
                      len_oovs,
                      x_padding_masks):
        """Function which returns a summary by always picking
           the highest probability option conditioned on the previous word.

        Args:
            encoder_input (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            max_oovs (int): Number of out-of-vocabulary tokens.

        Returns:
            summary (list): The token list of the result summary.
        """
        ###########################################
        #          TODO: module 4 task 1          #
        ###########################################

        # Get encoder output and states.
        encoder_output, encoder_states = self.model.encoder(replace_oovs(encoder_input, self.vocab), self.model.decoder.embedding)

        # Initialize decoder's hidden states with encoder's hidden states.
        decoder_states = self.model.reduce_state(encoder_states)

        # Initialize decoder's input at time step 0 with the SOS token.
        decoder_input_t = torch.ones(1) * self.vocab.SOS
        decoder_input_t = decoder_input_t.to(self.DEVICE, dtype=torch.int64)
        summary = [self.vocab.SOS]
        
        coverage_vector = torch.zeros((encoder_input.shape)).to(self.DEVICE)

        # Generate hypothesis with maximum decode step.
        while int(decoder_input_t.item()) != (self.vocab.EOS) \
                and len(summary) < max_sum_len:

            context_vector, attention_weights, coverage_vector = self.model.attention(decoder_states, encoder_output, x_padding_masks, coverage_vector)

            p_vocab, decoder_states, p_gen = self.model.decoder(decoder_input_t, decoder_states, encoder_output, context_vector)
            
            if 'baseline' in config.model_name:
                final_dist = p_vocab
            else:
                final_dist = self.model.get_final_distribution(encoder_input,
                                                               p_gen,
                                                               p_vocab,
                                                               attention_weights,
                                                               torch.max(len_oovs))

            # Get next token with maximum probability.
            decoder_input_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            decoder_word_idx = decoder_input_t.item()
            summary.append(decoder_word_idx)
            # Replace the indexes of OOV words with the index of UNK token
            # to prevent index-out-of-bound error in the decoder.
            decoder_input_t = replace_oovs(decoder_input_t, self.vocab)

        return summary

#     @timer('best k')
    def best_k(self, beam, k, encoder_output, x_padding_masks, x, len_oovs):
        """Get best k tokens to extend the current sequence at the current time step.

        Args:
            beam (untils.Beam): The candidate beam to be extended.
            k (int): Beam size.
            encoder_output (Tensor): The lstm output from the encoder.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.
            x (Tensor): Source token ids.
            len_oovs (Tensor): Number of oov tokens in a batch.

        Returns:
            best_k (list(Beam)): The list of best k candidates.

        """
        ###########################################
        #          TODO: module 4 task 2.2        #
        ###########################################

        # use decoder to generate vocab distribution for the next token
#         decoder_input_t = torch.tensor(beam.tokens[-1]).reshape(1, 1)
        decoder_input_t = torch.ones(1) * beam.tokens[-1]
        decoder_input_t = decoder_input_t.long().to(self.DEVICE)

        # Get context vector from attention network.
        context_vector, attention_weights, coverage_vector = \
            self.model.attention(beam.decoder_states,
                                 encoder_output,
                                 x_padding_masks,
                                 beam.coverage_vector)

        # Replace the indexes of OOV words with the index of UNK token
        # to prevent index-out-of-bound error in the decoder.
        decoder_input_t = replace_oovs(decoder_input_t, self.vocab)
#         print(decoder_input_t)
#         print('decoder input size', decoder_input_t.shape)
        p_vocab, decoder_states, p_gen = self.model.decoder(decoder_input_t, beam.decoder_states, encoder_output, context_vector)

#         print(p_vocab.shape)
#         print(p_vocab)
        # Calculate log probabilities.
#         log_probs = torch.log(p_vocab)

        if 'baseline' in config.model_name:
            final_dist = p_vocab
            
        else:
            final_dist = self.model.get_final_distribution(x,
                                                       p_gen,
                                                       p_vocab,
                                                       attention_weights,
                                                       torch.max(len_oovs))
        
        # Calculate log probabilities.
        log_probs = torch.log(final_dist)
#         print(log_probs.squeeze()[2])
        # Filter forbidden tokens.
#         if len(beam.tokens) == 1:
#             forbidden_ids = [
#                 self.vocab[u"这"],
#                 self.vocab[u"此"],
#                 self.vocab[u"采用"],
#                 self.vocab[u"，"],
#                 self.vocab[u"。"],
#             ]
#             log_probs[forbidden_ids] = -float('inf')
            
        # EOS token penalty
        log_probs.squeeze()[self.vocab.EOS] *= config.gamma * x.size()[1] / len(beam.tokens)
        
        log_probs.squeeze()[self.vocab.UNK] = -float('inf')
        
#         print(log_probs.shape)

        # Get top k tokens and the corresponding logprob.
        topk_probs, topk_idx = torch.topk(log_probs, k, dim=1)

#         print(topk_probs.shape)
#         print(topk_idx)
        # Extend the current hypo with top k tokens, resulting k new hypos.
        best_k = []
        for i in range(k):
            best_k.append(beam.extend(
                topk_idx[0][i].item(),
                topk_probs[0][i].item(),
                decoder_states,
                coverage_vector
            ))
            
#         for idx in topk_idx.tolist()[0]:
#             best_k.append(beam.extend(idx,
#                                       log_probs[idx],
#                                       decoder_states))
#         print(topk_idx.tolist()[0])
#         for i in topk_idx.tolist()[0]:
#             print(i)
            
#         best_k = [beam.extend(x,
#                   log_probs[x],
#                   decoder_states) for x in topk_idx.tolist()[0]]

        return best_k

    def beam_search(self,
                    encoder_input,
                    max_sum_len,
                    beam_width,
                    len_oovs,
                    x_padding_masks):
        """Using beam search to generate summary.

        Args:
            encoder_input (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            beam_width (int): Beam size.
            max_oovs (int): Number of out-of-vocabulary tokens.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.

        Returns:
            result (list(Beam)): The list of best k candidates.
        """
        ###########################################
        #          TODO: module 4 task 2.3        #
        ###########################################

        # run body_sequence input through encoder
        encoder_output, encoder_states = self.model.encoder(replace_oovs(encoder_input, self.vocab), self.model.decoder.embedding)
        
        coverage_vector = torch.zeros((encoder_input.shape)).to(self.DEVICE)

        # initialize decoder states with encoder forward states
        decoder_states = self.model.reduce_state(encoder_states)

        # initialize the hypothesis with a class Beam instance.
#         attention_weights = 
        init_beam = Beam(
            [self.vocab.SOS],
            [0],
            decoder_states,
            coverage_vector
        )

        # get the beam size and create a list for stroing current candidates
        # and a list for completed hypothesis
        k = beam_width
        curr, completed = [init_beam], []

        # use beam search for max_sum_len (maximum length) steps
        for _ in range(max_sum_len):
            # get k best hypothesis when adding a new token

            topk = []
            for beam in curr:
                # When an EOS token is generated, add the hypo to the completed
                # list and decrease beam size.
                if beam.tokens[-1] == self.vocab.EOS:
                    completed.append(beam)
                    k -= 1

                for can in self.best_k(beam,
                                       k,
                                       encoder_output,
                                       x_padding_masks,
                                       encoder_input,
                                       torch.max(len_oovs)):
                    # Using topk as a heap to keep track of top k candidates.
                    # Using the sequence scores of the hypos to campare
                    # and object ids to break ties.
#                     print(can.tokens)
#                     print(can.log_probs)
                    item = (can.seq_score(), id(can), can)
                    add2heap(topk, item, k)

            curr = [item[2] for item in topk]
            # stop when there are enough completed hypothesis
            if len(completed) == k:
                break
        # When there are not engouh completed hypotheses,
        # take whatever when have in current best k as the final candidates.
        completed += curr
        # sort the hypothesis by normalized probability and choose the best one
#         hq_completed = heapq.heapify(completed)
# [heappop(h) for i in range(len(h))]
        result = [heapq.heappop(completed) for i in range(len(completed))][-1].tokens
        return result

    @timer(module='doing prediction')
    def predict(self, text, tokenize=True, beam_search=False):
        """Generate summary.

        Args:
            text (str or list): Source.
            tokenize (bool, optional):
                Whether to do tokenize or not. Defaults to True.
            beam_search (bool, optional):
                Whether to use beam search or not.
                Defaults to True (means using greedy search).

        Returns:
            str: The final summary.
        """
        # Do tokenization if the input is raw text.
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)
#         max_oovs = len(oov)
        len_oovs = torch.tensor([len(oov)]).to(self.DEVICE)
        x_copy = replace_oovs(x, self.vocab)
        x_copy = x_copy.unsqueeze(0)
        x_padding_masks = torch.ne(x_copy, 0).byte().float()
        if beam_search:  # Use beam search to decode.
            summary = self.beam_search(x_copy,
                                       max_sum_len=config.max_dec_steps,
                                       beam_width=config.beam_size,
                                       len_oovs=len_oovs,
                                       x_padding_masks=x_padding_masks)
        else:  # Use greedy search to decode.
            summary = self.greedy_search(x_copy,
                                         max_sum_len=config.max_dec_steps,
                                         len_oovs=len_oovs,
                                         x_padding_masks=x_padding_masks)
        summary = outputids2words(summary,
                                  oov,
                                  self.vocab)
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()


if __name__ == "__main__":
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))
    # Randomly pick a sample in test set to predict.
    with open(config.test_data_path, 'r') as test:
        picked = random.Random(1008).choice(list(test))
        source, ref = picked.strip().split('<sep>')

    greedy_prediction = pred.predict(source.split(),  beam_search=False)
    beam_prediction = pred.predict(source.split(),  beam_search=True)

    print('source: ', source)
    print('greedy: ', greedy_prediction)
    print('beam: ', beam_prediction)
    print('ref: ', ref)
