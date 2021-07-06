#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: lpx, jby
@Date: 2020-07-13 11:00:51
@LastEditTime: 2020-07-18 16:47:58
@LastEditors: Please set LastEditors
@Description: Define the model.
@FilePath: /JD_project_2/baseline/model/model.py
'''


import os
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
import config

from utils import *


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 rnn_drop: float = 0):
        ###########################################
        #          TODO: module 2 task 1.1        #
        ###########################################
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, bidirectional=True, dropout=rnn_drop, batch_first=True)

    def forward(self, x, decoder_embedding):
        """Define forward propagation for the endoer.

        Args:
            x (Tensor): The input samples as shape (batch_size, seq_len).

        Returns:
            output (Tensor):
                The output of lstm with shape
                (batch_size, seq_len, 2 * hidden_units).
            hidden (tuple):
                The hidden states of lstm (h_n, c_n).
                Each with shape (2, batch_size, hidden_units)
        """
        ###########################################
        #          TODO: module 2 task 1.2        #
        ###########################################
#         if config.weight_tying:
#             embeded = decoder_embedding(x)
#         else:
        embeded = self.embedding(x)  # batch_size, seq_len, embed_size
        
        output, hidden = self.lstm(embeded)  # output -> (batch_size, seq_len, 2 * hidden_size)
                                             # h_n -> (2, batch_size, hidden_size)
                                             # c_n -> (2, batch_size, hidden_size)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        ###########################################
        #          TODO: module 2 task 3.1        #
        ###########################################
        self.Wh = nn.Linear(2*hidden_units, 2*hidden_units, bias=False)
        self.Ws = nn.Linear(2*hidden_units, 2*hidden_units)
        self.v = nn.Linear(2*hidden_units, 1, bias=False)
        
#         if config.coverage is True:
        self.w_c = nn.Linear(1, 2*hidden_units, bias=False)

    def forward(self,
                decoder_states,
                encoder_output,
                x_padding_masks,
                coverage_vector):
        """Define forward propagation for the attention network.

        Args:
            decoder_states (tuple):
                The hidden states from lstm (h_n, c_n) in the decoder,
                each with shape (1, batch_size, hidden_units)
            encoder_output (Tensor):
                The output from the lstm in the decoder with
                shape (batch_size, seq_len, hidden_units).
            x_padding_masks (Tensor):
                The padding masks for the input sequences
                with shape (batch_size, seq_len).
            coverage_vector (Tensor):
                The coverage vector from last time step.
                with shape (batch_size, seq_len).

        Returns:
            context_vector (Tensor):
                Dot products of attention weights and encoder hidden states.
                The shape is (batch_size, 2*hidden_units).
            attention_weights (Tensor): The shape is (batch_size, seq_length).
            coverage_vector (Tensor): The shape is (batch_size, seq_length).
        """
        ###########################################
        #          TODO: module 2 task 3.2        #
        ###########################################

        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
#         print('h_dec size: ', h_dec.shape)
#         print('c_dec size: ', c_dec.shape)
        # (1, batch_size, 2*hidden_units)
        s_t = torch.cat([h_dec, c_dec], dim=2)
#         print('s_t size: ', s_t.shape)
        # (batch_size, 1, 2*hidden_units)
#         s_t = s_t.squeeze(0).unsqueeze(1)
        s_t = s_t.transpose(0, 1)
        # (batch_size, seq_length, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()

        # calculate attention scores
        # Equation(11).
        # Wh h_* (batch_size, seq_length, 2*hidden_units)
        encoder_features = self.Wh(encoder_output)
        # Ws s_t (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)
        # (batch_size, seq_length, 2*hidden_units)
        att_inputs = encoder_features + decoder_features
        
        if config.coverage:
            att_inputs = att_inputs + self.w_c(coverage_vector.unsqueeze(2))
        
        # (batch_size, seq_length, 1)
        score = self.v(F.tanh(att_inputs))
        
        # (batch_size, seq_length)
        attention_weights = F.softmax(score, dim=1).squeeze(2)
        attention_weights = attention_weights * x_padding_masks
        
        # Normalize attention weights after excluding padded positions.
        normalization_factor = attention_weights.sum(1, keepdim=True)
        attention_weights = attention_weights / normalization_factor
        
        # (batch_size, 1, 2*hidden_units)
        attention_weights = attention_weights.unsqueeze(1)
        context_vector = torch.bmm(attention_weights, encoder_output)
        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)
        
        attention_weights = attention_weights.squeeze(1)
        
        # update coverage vector
        if config.coverage:
            coverage_vector = coverage_vector + attention_weights

        return context_vector, attention_weights, coverage_vector


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 enc_hidden_size=None,
                 is_cuda=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        ###########################################
        #          TODO: module 2 task 2.1        #
        ###########################################

        self.lstm = nn.LSTM(embed_size, self.hidden_size, batch_first=True)

        self.W1 = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, self.vocab_size)
        
        if config.pointer:
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)

    def forward(self, decoder_input, decoder_states, encoder_output,
                context_vector):
        """Define forward propagation for the decoder.

        Args:
            decoder_input (Tensor):
                The input of the decoder x_t of shape (batch_size, 1).
            decoder_states (tuple):
                The hidden states(h_n, c_n) of the decoder from last time step.
                The shapes are (1, batch_size, hidden_units) for each.
            encoder_output (Tensor):
                The output from the encoder of shape
                (batch_size, seq_length, 2*hidden_units).
            context_vector (Tensor):
                The context vector from the attention network
                of shape (batch_size,2*hidden_units).

        Returns:
            p_vocab (Tensor):
                The vocabulary distribution of shape (batch_size, vocab_size).
            docoder_states (tuple):
                The lstm states in the decoder.
                The shapes are (1, batch_size, hidden_units) for each.
        """

        ###########################################
        #          TODO: module 2 task 2.2        #
        ###########################################
        decoder_input = decoder_input.unsqueeze(1)
#         print('decoder input size', decoder_input.shape)

        decoder_emb = self.embedding(decoder_input)  # (batch_size, 1, embed_size)
        
#         print('decoder_emb size', decoder_emb.shape)
#         print('decoder_states size', len(decoder_states))
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)  # decoder_output -> (1, batch_size, hidden_size)
                                                                                 # decoder_states -> (1, batch_size, hidden_size), (1, batch_size, hidden_size)

        # concatenate context vector and decoder state
        # (batch_size, 3*hidden_units)
        decoder_output =  decoder_output.view(-1, self.hidden_size)  # Reshape decoder_output to align with context_vector
        concat_vector = torch.cat((decoder_output, context_vector), 1)

        # calculate vocabulary distribution
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)
        
        
        # (batch_size, vocab_size)
        if config.weight_tying:
            FF2_out = torch.mm(FF1_out, torch.t(self.embedding.weight))
        else:
            FF2_out = self.W2(FF1_out)
        
        
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)
        
        # concatenate h and c to get s_t and expand the dim of s_t
        h_t, c_t = decoder_states
        s_t = torch.cat([h_t, c_t], dim=2)
        
        
        p_gen = None
        if config.pointer is True:
            p_input = torch.cat([
                context_vector,
                s_t.squeeze(0),
                decoder_emb.squeeze(1)
            ], dim=-1)
            p_gen = torch.sigmoid(self.w_gen(p_input))

        return p_vocab, decoder_states, p_gen


class ReduceState(nn.Module):
    """
    Since the encoder has a bidirectional LSTM layer while the decoder has a
    unidirectional LSTM layer, we add this module to reduce the hidden states
    output by the encoder (merge two directions) before input the hidden states
    nto the decoder.
    """
    ###########################################
    #          TODO: module 2 task 5          #
    ###########################################

    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        """The forward propagation of reduce state module.

        Args:
            hidden (tuple):
                Hidden states of encoder,
                each with shape (2, batch_size, hidden_units).

        Returns:
            tuple:
                Reduced hidden states,
                each with shape (1, batch_size, hidden_units).
        """
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)
        return (h_reduced, c_reduced)
    
    
class Seq2seq(nn.Module):
    def __init__(
            self,
            v
    ):
        super(Seq2seq, self).__init__()
        self.v = v
        self.DEVICE = torch.device("cuda" if config.is_cuda else "cpu")
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(
            len(v),
            config.embed_size,
            config.hidden_size,
        )
        self.decoder = Decoder(len(v),
                               config.embed_size,
                               config.hidden_size,
                               )
        self.reduce_state = ReduceState()
        self.lambda_cov = torch.tensor(1.,
                                       requires_grad=False,
                                       device=self.DEVICE)

    def load_model(self):
        """Load saved model if there exits one.
        """        
        if (os.path.exists(config.encoder_save_name)):
            self.encoder = torch.load(config.encoder_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.reduce_state = torch.load(config.reduce_state_save_name)

    def forward(self, x, x_len, y, len_oovs, batch):
        """Define the forward propagation for the seq2seq model.

        Args:
            x (Tensor):
                Input sequences as source with shape (batch_size, seq_len)
            x_len (int): Sequence length of the current batch.
            y (Tensor):
                Input sequences as reference with shape (bacth_size, y_len)
            len_oovs (int):
                The number of out-of-vocabulary words in this sample.
            batch (int): The number of the current batch.

        Returns:
            batch_loss (Tensor): The average loss of the current batch.
        """

        ###########################################
        #          TODO: module 2 task 4          #
        ###########################################

        oov_token = torch.full(x.shape, self.v.UNK).long().to(config.DEVICE)
        x_copy = torch.where(x > self.v.size() - 1, oov_token, x)
        x_padding_masks = torch.ne(x, self.v.PAD).byte().float()
        encoder_output, encoder_states = self.encoder(x_copy)
        # Reduce encoder hidden states.
        decoder_states =  self.reduce_state(encoder_states)

        # Calculate loss for every step.
        step_losses = []
        for t in range(y.shape[1]-1):
            decoder_input_t = y[:, t] # x_t
            decoder_target_t = y[:, t+1]   # y_t
            # Get context vector from the attention network.
            context_vector, attention_weights = self.attention(decoder_states, encoder_output, x_padding_masks)
            
            # Get vocab distribution and hidden states from the decoder.
            p_vocab, decoder_states = self.decoder(decoder_input_t, decoder_states, encoder_output, context_vector)

            # Get the probabilities predict by the model for target tokens.
            target_probs = torch.gather(p_vocab, 1, decoder_target_t.unsqueeze(1))  # (batch_size, 1)
            target_probs = target_probs.squeeze(1)  # (batch_size,)
            
            # Apply a mask such that pad zeros do not affect the loss
            mask = torch.ne(decoder_target_t, self.v.PAD)  # (batch_size, )
            # Do smoothing to prevent getting NaN loss because of log(0).
            loss = - torch.log(target_probs + config.eps)  # (batch_size, )
            mask = mask.float()
            loss = loss * mask
            step_losses.append(loss)

        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)  # (batch_size, 1) -> sum of losses for all time steps per sample
        # get the non-padded length of each sequence in the batch
        seq_len_mask = torch.ne(y, self.v.PAD).byte().float()  # (batch_size, y_len)
        batch_seq_len = torch.sum(seq_len_mask, dim=1)  # (batch_size, )

        # get batch loss by dividing the loss of each batch
        # by the target sequence length and mean
#         print("sample_losses dtype", sample_losses.dtype)
#         print("batch_seq_len dtype", batch_seq_len.dtype)
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss

    
class PGN(nn.Module):
    def __init__(
            self,
            v
    ):
        super(PGN, self).__init__()
        self.v = v
        self.DEVICE = config.DEVICE
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(
            len(v),
            config.embed_size,
            config.hidden_size,
        )
        self.decoder = Decoder(len(v),
                               config.embed_size,
                               config.hidden_size,
                               )
        self.reduce_state = ReduceState()

    def load_model(self):

        if (os.path.exists(config.encoder_save_name)):
            print('Loading model: ', config.encoder_save_name)
            self.encoder = torch.load(config.encoder_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.reduce_state = torch.load(config.reduce_state_save_name)

        elif config.fine_tune:
            print('Loading model: ', '../saved_model/ss_pgn/encoder.pt')
            self.encoder = torch.load('../saved_model/ss_pgn/encoder.pt')
            self.decoder = torch.load('../saved_model/ss_pgn/decoder.pt')
            self.attention = torch.load('../saved_model/ss_pgn/attention.pt')
            self.reduce_state = torch.load('../saved_model/ss_pgn/reduce_state.pt')
            
            
    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights,
                               max_oov):
        """Calculate the final distribution for the model.

        Args:
            x: (batch_size, seq_len)
            p_gen: (batch_size, 1)
            p_vocab: (batch_size, vocab_size)
            attention_weights: (batch_size, seq_len)
            max_oov: (Tensor or int): The maximum sequence length in the batch.

        Returns:
            final_distribution (Tensor):
            The final distribution over the extended vocabualary.
            The shape is (batch_size, )
        """
        if not config.pointer:
            return p_vocab
        
        # clip p_gen
        p_gen = torch.clamp(p_gen, 0.001, 0.999)  # (batch_size, 1)
        
        # weighted p_vocab
        p_vocab_weighted = p_gen * p_vocab  # (batch_size, vocab_size)
        
        # weighted attention weights
#         print(attention_weights.shape)
        attention_weights_weighted = (1 - p_gen) * attention_weights  # (batch_size, seq_len)
        
        extend_vocab = torch.zeros((x.size()[0], max_oov)).float().to(self.DEVICE)
#         print(p_vocab_weighted.is_cuda)
#         print(extend_vocab.is_cuda)
        p_vocab_extend = torch.cat([p_vocab_weighted, extend_vocab], dim=1)  # (batch_size, vocab_size + max_oov)
        
        # x does not include UNKs
#         print(x.shape)
#         print(p_vocab_extend.shape)
#         print(attention_weights_weighted.shape) # dimension problem
        final_dist = p_vocab_extend.scatter_add_(dim=1, index=x, src=attention_weights_weighted)
        
        return final_dist
    
        
    def forward(self, x, x_len, y, len_oovs, batch, num_batches, teacher_forcing):
        """Define the forward propagation for the seq2seq model.

        Args:
            x (Tensor):
                Input sequences as source with shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor):
                Input sequences as reference with shape (bacth_size, y_len)
            len_oovs (Tensor):
                The numbers of out-of-vocabulary words for samples in this batch.
            batch (int): The number of the current batch.
            num_batches(int): Number of batches in the epoch.

        Returns:
            batch_loss (Tensor): The average loss of the current batch.
        """
#         oov_token = torch.full(x.shape, self.v.UNK).long().to(config.DEVICE)
#         x_copy = torch.where(x > self.v.size() - 1, oov_token, x)
        x_copy = replace_oovs(x, self.v)
        x_padding_masks = torch.ne(x, self.v.PAD).byte().float()
        encoder_output, encoder_states = self.encoder(x_copy, self.decoder.embedding)
        # Reduce encoder hidden states.
        decoder_states =  self.reduce_state(encoder_states)
        coverage_vector = torch.zeros((x.shape)).to(self.DEVICE)

        # Calculate loss for every step.
        step_losses = []
        decoder_input_t = y[:, 0]
        for t in range(y.shape[1]-1):
            if teacher_forcing:
                decoder_input_t = y[:, t] # x_t
                
            decoder_input_t = replace_oovs(decoder_input_t, self.v)  # this line is really important !!!!!
            
            decoder_target_t = y[:, t+1]   # y_t
            
            # Get context vector from the attention network.
            context_vector, attention_weights, coverage_vector = self.attention(decoder_states, encoder_output, x_padding_masks, coverage_vector)
            
            # Get vocab distribution and hidden states from the decoder.
            p_vocab, decoder_states, p_gen = self.decoder(decoder_input_t, decoder_states, encoder_output, context_vector)

            # Get final distribution
            final_dist = self.get_final_distribution(x, p_gen, p_vocab, attention_weights, torch.max(len_oovs))
            
            # update decoder_input_t
            decoder_input_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            
            if not config.pointer:
                decoder_target_t = replace_oovs(decoder_target_t, self.v)
            
            # Get the probabilities predicted by the model for target tokens.
            target_probs = torch.gather(final_dist, 1, decoder_target_t.unsqueeze(1))  # (batch_size, 1)
            target_probs = target_probs.squeeze(1)  # (batch_size,)
            
            # Apply a mask such that pad zeros do not affect the loss
            mask = torch.ne(decoder_target_t, self.v.PAD)  # (batch_size, )
            mask = mask.float()
            
            # Do smoothing to prevent getting NaN loss because of log(0).
            loss = - torch.log(target_probs + config.eps)  # (batch_size, )
            
            # coverage loss
            if config.coverage:
                loss_cov = torch.sum(torch.min(attention_weights, coverage_vector), dim=1)  # (batch_size,)
                loss = loss + config.LAMBDA * loss_cov
                
            loss = loss * mask
            step_losses.append(loss)

        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)  # (batch_size, 1) -> sum of losses for all time steps per sample
        # get the non-padded length of each sequence in the batch
        seq_len_mask = torch.ne(y, self.v.PAD).byte().float()  # (batch_size, y_len)
        batch_seq_len = torch.sum(seq_len_mask, dim=1)  # (batch_size, )

        # get batch loss by dividing the loss of each batch
        # by the target sequence length and mean
#         print("sample_losses dtype", sample_losses.dtype)
#         print("batch_seq_len dtype", batch_seq_len.dtype)
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss
        