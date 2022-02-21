''' Define the Transformer model '''
import torch, os
import torch.nn as nn
import numpy as np
import time
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.modeling import BertModel,BertEmbeddings
from transformer.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import  torch.nn.functional  as F
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
import json

INF = 1e10
EPSILON = 1e-10

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

#############################################################double decoder
class DecoderLayer_stage(nn.Module):
    ''' Compose with three layers''' 
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer_stage, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc2_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
 
    def forward(self, dec_input, enc_output, stage_2, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None, dec_stage_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask 
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask
        dec_output, stage_2_attn = self.enc2_attn(dec_output, stage_2, stage_2, mask=dec_stage_attn_mask)
        #dec_output *= non_pad_mask 
        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask
        return dec_output, dec_slf_attn, dec_enc_attn, stage_2_attn

class Decoder_bert_stage_2(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, len_max_seq, d_word_vec,
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
            DecoderLayer_stage(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.pointer_layer = PointerDecoder(d_model,d_model,dropout=0.2)

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, triples_words, enc_triples,enc_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list, stage_2_attn_list = [], [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)
        dec_enc_attn_mask2 = get_attn_key_pad_mask(seq_k=triples_words, seq_q=tgt_seq)

        # -- Forward
        #dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        dec_output = self.embedding(tgt_seq)
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn, stage_2_attn = dec_layer(
                dec_output, enc_output, enc_triples,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
                dec_stage_attn_mask=dec_enc_attn_mask2)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]
                stage_2_attn_list += [stage_2_attn]
        enc_padding = enc_mask.data == Constants.PAD
        dec_output = self.pointer_layer(dec_output,enc_output,atten_mask = enc_padding)

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list, stage_2_attn_list
        return dec_output,


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


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


class Feature_enrich(nn.Module):
    def __init__(self, d_model, d_feature, d_inner, d_k, d_v, dropout):
        super().__init__()
        self.highway = Highway(d_model+d_feature, 2, torch.nn.functional.sigmoid)
        self.tanh1 = nn.Tanh()
        #atten
        self.atten_1 = EncoderLayer(d_model=d_model+d_feature,
                                    d_inner=d_inner,
                                    n_head=1,
                                    d_k=d_k,
                                    d_v=d_v,
                                    dropout=dropout)
        self.to_origin_size = nn.Linear(d_model+d_feature, d_model)
    
    def forward(self, src_seq, input, feature):
        cat_out = torch.cat([input, feature], dim=-1)
        output = self.highway(cat_out)
        output = self.tanh1(output)
        #atten
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)
        output,_ = self.atten_1(output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        output = self.to_origin_size(output)
        return output


class Transformer_bert(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, n_entity_vocab, len_max_seq, len_max_tgt,bert_model='uncased_L-12_H-768_A-12',
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True,
            entity_relation_emb=None,
            num_entities=0,
            entity_hidden_size=100):

        super().__init__()
        self.vocab_size = n_tgt_vocab
        self.entity_hidden_size = entity_hidden_size
        # self.encoder = Encoder(
            # n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            # d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            # n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            # dropout=dropout)

        ###################################add by wy. for cmm-master
        self.entity_vocab_size = n_entity_vocab # entity_vocab_size 包含于 vocab_size
        self.entity_embed = nn.Embedding(num_entities, entity_hidden_size)
        print(self.entity_embed)
        print("load transE embedding")
        self.entity_embed.weight.data.copy_(torch.from_numpy(entity_relation_emb))

        self.encoder = BertModel.from_pretrained(bert_model,cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1)))
        decoder_embedding = build_decoder_embedding(self.encoder)
        
        # tmp: enhance in the encode
        self.feature_enrich = Feature_enrich(d_model, entity_hidden_size*2, d_inner, d_k, d_v, dropout)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        #graph embedding
        #fact matching
        self.fact_mlp1 = nn.Linear(d_model+entity_hidden_size*2, 256)
        self.fact_mlp2 = nn.Linear(256, 1)
        self.head_tail_linear = nn.Linear(entity_hidden_size*2, entity_hidden_size)
        self.relation_linear = nn.Linear(entity_hidden_size, entity_hidden_size)
        self.m = nn.Tanh()
        self.softmax = nn.Softmax()

        # linear for decoder2
        self.triple_linear = nn.Linear(entity_hidden_size*2, d_model)

        self.decoder2 = Decoder_bert_stage_2(len_max_seq=len_max_tgt,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout,decoder_embedding=decoder_embedding)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        ##### entity vocab
        self.selector = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1), nn.Sigmoid())
        self.entity_word_prj = nn.Linear(d_model, self.entity_vocab_size, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        nn.init.xavier_normal_(self.entity_word_prj.weight)

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

    def decoder(self, tgt_seq, tgt_pos, src_seq, enc_output, fact_triples, src_mask):
        # fix for beam search mismatch err
        if enc_output.shape[0] < fact_triples.shape[0]:
            n_bz = enc_output.shape[0]
            fact_triples = fact_triples[:n_bz]
        dec_output, dec_slf_attn_list, dec_enc_attn_list, stage_2_attn_list = self.decoder2(tgt_seq, tgt_pos, src_seq, enc_output, src_seq, fact_triples, src_mask, return_attns=True)
        return dec_output, stage_2_attn_list

    def forward(self, src_seq, src_pos, src_mask, src_seg_ids, tgt_seq, tgt_pos, src_triples, entity_word_map=None, getencoder=False):     #src_turns_label, 

        if tgt_seq is not None:
            tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]
        enc_output, *_ = self.encoder(src_seq,token_type_ids=src_seg_ids,attention_mask=src_mask, output_all_encoded_layers=False)  # shape: [?, Nx, 512]

        #########add by wy , ccm-master
        triples_embedding = self.entity_embed(src_triples) #shape: [?, Nf, 3, 100]

        # Static Graph Attention
        head, relation, tail = torch.split(triples_embedding, [1] * 3, dim=2) # shape: [?, Nf, 1, 100] * 3
        head = torch.squeeze(head) # shape: [?, Nf, 100]
        relation = torch.squeeze(relation) # shape: [?, Nf, 100]
        tail = torch.squeeze(tail) # shape: [?, Nf, 100]
        head_tail = torch.cat([head, tail], dim=-1) # shape: [?, Nf, 200]
        head_tail_transformed = self.head_tail_linear(head_tail) # shape:[?, Nf, 100]
        head_tail_transformed = self.m(head_tail_transformed) 
        relation_transformed = self.relation_linear(relation) # shape:[?, Nf, 100]
        e_weight = torch.sum(relation_transformed * head_tail_transformed, dim = -1) # shape:[?, Nf, 1]
        alpha_weight = self.softmax(e_weight) # shape:[?, Nf, 1]
        graph_embed = alpha_weight.unsqueeze(2) * head_tail # shape: [?, Nf, 200]

        fact_matching, enfact_weight = self.fact_match_cal(enc_output, graph_embed)

        feature_enc_output = self.feature_enrich(src_seq, enc_output, fact_matching)
        fact_triples = self.triple_linear(fact_matching)

        if getencoder:
            return feature_enc_output, fact_triples, alpha_weight

        dec_output, *_ = self.decoder2(tgt_seq, tgt_pos, src_seq, feature_enc_output, src_seq, fact_triples, src_mask)  ###    ##enc_output2    triple state
        #print(dec_output)
        cont_ans_outputs, cont_ans_weight, vocab_pointer_switches = dec_output
        probs, select_prob = self.probs(cont_ans_outputs, vocab_pointer_switches, cont_ans_weight, src_seq, entity_word_map)
        logits = probs.float()
        lprobs = torch.log(logits)
        select_logits = select_prob.float()
        return lprobs.view(-1,lprobs.size(-1)), select_logits.view(-1, select_logits.size(-1))


    def fact_match_cal(self, enc_output, mean_triples):
        Nf = mean_triples.shape[-2]
        Nx = enc_output.shape[-2]
        expand_enc = enc_output.expand([Nf] + list(enc_output.shape)).permute(1,2,0,3) # shape: [?, Nx, Nf, 512]
        expand_tri = mean_triples.expand([Nx] + list(mean_triples.shape)).permute(1,0,2,3) # shape: [?, Nx, Nf, 200]
        expand_cat = torch.cat([expand_enc, expand_tri], dim=-1) # shape: [?, Nx, Nf, 712]
        cat_out = self.tanh(self.fact_mlp1(expand_cat)) # shape: [?, Nx, Nf, 256]
        weight = self.fact_mlp2(cat_out) # shape: [?, Nx, Nf, 1]
        weight = self.softmax(weight) 
        #sum_weight = torch.sum(weight, dim=-2)
        #print(sum_weight)
        #print(sum_weight.shape)
        #time.sleep(5)
        sum_output = torch.sum(weight * expand_tri, dim=2)/torch.sum(weight, dim=2) # shape: [?, Nx, 200]
        return sum_output, weight


    def entity_difussion_cal(self, enc_output, fact_matching, entity_embedding):
        pre_cat = torch.cat([enc_output, fact_matching], dim=-1) # shape:[?, Nx, 612]
        Ne = entity_embedding.shape[-2]
        Nx = enc_output.shape[-2]
        expand_pre = pre_cat.expand([Ne] + list(pre_cat.shape)).permute(1,2,0,3) # shape: [?, Nx, Ne, 612]
        expand_tri = entity_embedding.expand([Nx] + list(entity_embedding.shape)).permute(1,0,2,3) # shape: [?, Nx, Ne, 100]
        expand_cat = torch.cat([expand_pre, expand_tri], dim=-1) # shape: [?, Nx, Ne, 712]
        cat_out = self.tanh(self.entity_mlp1(expand_cat)) # shape: [?, Nx, Ne, 256]
        weight = self.softmax(self.entity_mlp2(cat_out)) # shape: [?, Nx, Ne, 1]
        # TODO: topn limit if needed
        sum_output = torch.sum(weight * expand_tri, dim=2)/torch.sum(weight, dim=2) # shape: [?, Nx, 100]
        return sum_output

    ####### modified
    def probs(self, outputs, vocab_pointer_switches, context_question_weight, context_question_indices, entity_word_map, oov_to_limited_idx=None):

        size = list(outputs.size())

        #self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        #self.selector = nn.Linear(d_model, 1)
        #self.entity_word_prj = nn.Linear(d_model, n_entity_vocab, bias=False)

        # copynet src
        size[-1] = self.vocab_size
        scores = self.tgt_word_prj(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim()-1)
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim()-1, context_question_indices.unsqueeze(1).expand_as(context_question_weight),
            (1 - vocab_pointer_switches).expand_as(context_question_weight) * context_question_weight)

        # entitynet
        size[-1] = self.entity_vocab_size
        scores = self.entity_word_prj(outputs.view(-1, outputs.size(-1))).view(size)
        entity_vocab = F.softmax(scores, dim=scores.dim()-1) #[?, Nt, Ve]
        vocab_entity_switches = self.selector(outputs)
        entity_scaled_p_vocab = vocab_entity_switches.expand_as(scaled_p_vocab) * scaled_p_vocab # selecor * metrix
        entity_scaled_p_vocab.scatter_add_(entity_scaled_p_vocab.dim()-1, entity_word_map.long().unsqueeze(0).expand_as(entity_vocab),
            (1 - vocab_entity_switches).expand_as(entity_vocab) * entity_vocab)
        return entity_scaled_p_vocab, vocab_entity_switches

    def predict(self, outputs, vocab_pointer_switches, context_question_weight, context_question_indices, entity_word_map, oov_to_limited_idx=None):

        size = list(outputs.size())

        #self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        #self.selector = nn.Linear(d_model, 1)
        #self.entity_word_prj = nn.Linear(d_model, n_entity_vocab, bias=False)

        # copynet src
        size[-1] = self.vocab_size
        scores = self.tgt_word_prj(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim()-1)
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim()-1, context_question_indices.expand_as(context_question_weight),
            (1 - vocab_pointer_switches).expand_as(context_question_weight) * context_question_weight)

        # entitynet
        size[-1] = self.entity_vocab_size
        scores = self.entity_word_prj(outputs.view(-1, outputs.size(-1))).view(size)
        entity_vocab = F.softmax(scores, dim=scores.dim()-1) #[?, Nt, Ve]
        vocab_entity_switches = self.selector(outputs)
        entity_scaled_p_vocab = vocab_entity_switches.expand_as(scaled_p_vocab) * scaled_p_vocab # selecor * metrix
        entity_scaled_p_vocab.scatter_add_(entity_scaled_p_vocab.dim()-1, entity_word_map.long().unsqueeze(0).expand_as(entity_vocab),
            (1 - vocab_entity_switches).expand_as(entity_vocab) * entity_vocab)

        lprobs = entity_scaled_p_vocab.float()
        lprobs = lprobs.log()
        return lprobs, vocab_pointer_switches, vocab_entity_switches


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
        #shape: torch.Size([8, 61, 768]), torch.Size([8, 61, 102])
        vocab_pointer_switches = self.vocab_pointer_switch(torch.cat([input, context_question_outputs], -1))
        # selector: torch.Size([8, 61, 1])
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
