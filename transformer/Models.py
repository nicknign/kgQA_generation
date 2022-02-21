''' Define the Transformer model '''
import torch, os
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.modeling import BertModel,BertEmbeddings
from transformer.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import  torch.nn.functional  as F

INF = 1e10
EPSILON = 1e-10

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def build_decoder_embedding(encoder):
    decoder_embedding = BertEmbeddings(encoder.config)
    encoder_embedding = encoder.embeddings
    decoder_embedding.word_embeddings.weight.data.copy_(encoder_embedding.word_embeddings.weight.data)
    decoder_embedding.position_embeddings.weight.data.copy_(encoder_embedding.position_embeddings.weight.data)
    decoder_embedding.token_type_embeddings.weight.data.copy_(encoder_embedding.token_type_embeddings.weight.data)
    decoder_embedding.LayerNorm.weight.data.copy_(encoder_embedding.LayerNorm.weight.data)
    decoder_embedding.LayerNorm.bias.data.copy_(encoder_embedding.LayerNorm.bias.data)
    return decoder_embedding

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

##########################################
class Encoder_answerfeature(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)
        self.token_type_embeddings = nn.Embedding(2, d_word_vec, padding_idx=0)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos,ans_seq,ans_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos) + self.token_type_embeddings(ans_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

#################################################################
class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class Decoder_bert(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, decoder_embedding=None):

        super().__init__()
        # n_position = len_max_seq + 1
        #
        # self.tgt_word_emb = nn.Embedding(
        #     n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        #
        # self.position_enc = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
        #     freeze=True)
        self.embedding = decoder_embedding
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.pointer_layer = PointerDecoder(d_model,d_model,dropout=0.2)

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output,enc_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        #dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        dec_output = self.embedding(tgt_seq)
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]
        enc_padding = enc_mask.data == Constants.PAD
        dec_output = self.pointer_layer(dec_output,enc_output,atten_mask = enc_padding)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))



class Transformer_bert(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq, len_max_tgt,bert_model='uncased_L-12_H-768_A-12',
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()
        self.vocab_size = n_tgt_vocab
        # self.encoder = Encoder(
            # n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            # d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            # n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            # dropout=dropout)

        self.encoder = BertModel.from_pretrained(bert_model,cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1)))
        decoder_embedding = build_decoder_embedding(self.encoder)


        ## LINFENG new_loss
        self.head_classify = nn.Linear(d_model, 1)
        ## END

        self.decoder = Decoder_bert(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_tgt,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout,decoder_embedding=decoder_embedding)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight



    def forward(self, src_seq, src_pos, src_mask, tgt_seq, tgt_pos, ans_seq, src_head_of_turns_label):     #src_turns_label, 

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq,token_type_ids=ans_seq,attention_mask=src_mask, output_all_encoded_layers=False)

        ### LINFENG new_loss
        head_logits = self.head_classify(enc_output)
        ###END

        # print('0')
        # print(enc_output)
        # print('1')
        # print(tgt_seq)
        # print('2')
        # print(tgt_pos)
        # print('3')
        # print(src_seq)
        ###############################################
        #shorter_src_mask = 1-torch.eq(src_mask,ans_seq)
        ###############################################
        #dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, shorter_src_mask)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output, src_mask)
        cont_ans_outputs, cont_ans_weight, vocab_pointer_switches = dec_output
        #seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
        probs = self.probs(self.tgt_word_prj,cont_ans_outputs,vocab_pointer_switches, cont_ans_weight,src_seq)
        #print(probs.shape)
        #return seq_logit.view(-1, seq_logit.size(2))
        logits = probs.float()
        lprobs = torch.log(logits)

        return lprobs.view(-1,lprobs.size(-1)), head_logits      ##  LINFENG

    def probs(self, generator, outputs, vocab_pointer_switches, context_question_weight, context_question_indices, oov_to_limited_idx=None):

        size = list(outputs.size())

        size[-1] = self.vocab_size
        #if self.args.share_decoder_input_output_embed and self.args.reduce_dim > 0:
        #    outputs = self.project(outputs)
        # B * Ta * V
        scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim()-1)
        #p_vocab = scores
        # B * Ta * V
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = self.vocab_size # + len(oov_to_limited_idx) TODO: add the oov
        if self.vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim()-1)

        # p_context_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim()-1, context_question_indices.unsqueeze(1).expand_as(context_question_weight),
            (1 - vocab_pointer_switches).expand_as(context_question_weight) * context_question_weight)

        return scaled_p_vocab

    def predict(self, generator, outputs, vocab_pointer_switches, context_question_weight, context_question_indices, oov_to_limited_idx=None):

        size = list(outputs.size())

        size[-1] = self.vocab_size
        #if self.args.share_decoder_input_output_embed and self.args.reduce_dim > 0:
        #    outputs = self.project(outputs)
        # B * Ta * V
        scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim()-1)
        #p_vocab = scores
        # B * Ta * V
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = self.vocab_size # + len(oov_to_limited_idx) TODO: add the oov
        if self.vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim()-1)

        # p_context_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim()-1, context_question_indices.expand_as(context_question_weight),
            (1 - vocab_pointer_switches).expand_as(context_question_weight) * context_question_weight)
        lprobs = scaled_p_vocab.float()
        #print(scaled_p_vocab.max(-1))
        lprobs = lprobs.log()
        #print(lprobs.max(-1))
        return lprobs


##############################
class Transformer_answerfeature(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq, len_max_tgt,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder = Encoder_answerfeature(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_tgt,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos,ans_seq,ans_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos,ans_seq,ans_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
#########################################################

class PointerDecoder(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.0, num_layers=1):
        super().__init__()
        self.d_hid = d_hid
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.context_question_attn = DecoderAttention(d_hid, dot=True)  # TJ
        self.vocab_pointer_switch = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.d_hid + d_in, 1), nn.Sigmoid()) # XD

    # input = B * T * C
    def forward(self, input, context_question, atten_mask):

        context_question_outputs, context_question_weight = self.context_question_attn(input, context_question, atten_mask=atten_mask)
        vocab_pointer_switches = self.vocab_pointer_switch(torch.cat([input, context_question_outputs], -1))
        context_question_outputs = self.dropout(context_question_outputs)

        return context_question_outputs, context_question_weight, vocab_pointer_switches

class DecoderAttention(nn.Module): #TJ
    def __init__(self, dim, dot=False):
        super().__init__()
        if not dot:
            self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(2 * dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None
        self.dot = dot

    def applyMasks(self, context_mask):
        self.context_mask = context_mask

    def forward(self, input, context, atten_mask):
        if not self.dot:
            targetT = self.linear_in(input)  # B x Ta x C
        else:
            targetT = input

        context_scores = torch.bmm(targetT, context.transpose(1, 2))
        #print('context_scores: ',context_scores.shape)
        context_scores.masked_fill_(atten_mask.unsqueeze(1), -float('inf'))
        context_weight = F.softmax(context_scores, dim=-1) + EPSILON
        context_atten = torch.bmm(context_weight, context)

        combined_representation = torch.cat([input, context_atten], 2)
        output = self.tanh(self.linear_out(combined_representation))

        return output, context_weight
