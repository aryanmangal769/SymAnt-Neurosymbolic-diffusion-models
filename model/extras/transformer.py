"""
FUTR Transformer class.

Copy-paste from github.com/facebookresearch/detr/blob/main/models/transformer.py with modifications.

"""

import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from einops import repeat, rearrange, einsum
import copy
from typing import Optional, List
import pickle
from model.extras.mha import MultiheadAttention

from graph_modules.gsnn.gsnn import GSNN
from graph_modules.graph.graph import Graph
from graph_modules.gsnn.gsnn_forward import get_context_vectors
from graph_modules.gat.gatv2 import ModifiedGATv2
from graph_modules.gat.gat_forward import get_node_representations
from graph_modules.gat.video_enc import VideoEncoder
from graph_modules.graph_utils import merge_graphs

from model.extras.weight_matrix import KnowledgeWeightingModel
import math
import pdb
import torch
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba_simple import  Block
from functools import partial
from dataclasses import dataclass, field
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from model.extras.position import PositionalEncoding
import pdb

class Diffusion(nn.Module):

    def __init__(self, args, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        self.args = args
        self.d_head = d_head = d_model // nhead
        self.T = args.T

        if args.kg_attn == True or args.kg_init == True: 
            if args.graph_merging: 
                args.vocab_size = 500   #Setting a high vocab size becauce number of nodes vary for in merged graphs.

            self.knowledge_weighting_encoder_model = KnowledgeWeightingModel(args)
            self.knowledge_weighting_decoder_model = KnowledgeWeightingModel(args)

            encoder_layer = TransformerEncoderLayer(args, d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,
                                                    self.knowledge_weighting_encoder_model)

            decoder_layer = TransformerDecoderLayer(args, d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,
                                                    self.knowledge_weighting_decoder_model)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            if args.mamba:
                self.encoder = MixerModel( d_model =d_model, n_layer= num_encoder_layers)
            else:
                self.encoder = TransformerEncoder(args, encoder_layer, num_encoder_layers, encoder_norm)




            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(args, decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        else:
            encoder_layer = TransformerEncoderLayer(args,d_model, nhead, dim_feedforward,dropout, activation, normalize_before)
            # encoder_layer = MambaBlock(args,d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
            # encoder_layer = MambaBlock2(args,d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            if args.mamba:
                self.encoder = MixerModel( d_model =d_model, n_layer= num_encoder_layers)
            else:
                self.encoder = TransformerEncoder(args, encoder_layer, num_encoder_layers, encoder_norm)

            decoder_layer = TransformerDecoderLayer(args, d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(args, decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.device = torch.device('cuda')

        if args.kg_attn == True or args.kg_init == True:
            self.kg = Graph()
            self.kg = pickle.load(open('./datasets/graph_kitchen.pkl', 'rb'))
            # self.graph.getGlobalAdjacencyMat()

            if args.use_gsnn:
                self.gsnn_net = GSNN(args)
                self.gsnn_net = self.gsnn_net.to(torch.device(self.device))

                if args.condition_propagation:
                    self.video_encoder = VideoEncoder(input_size=args.hidden_dim, hidden_size=args.hidden_dim//4, 
                                                num_layers=2, output_size=args.condition_propagation_dim, max_len=1512)
                    self.video_encoder = self.video_encoder.to(torch.device(self.device))
            
            else:
                self.gat = ModifiedGATv2(args, in_features=args.state_dim*2, n_hidden=args.state_dim, 
                                        n_heads=args.state_dim, dropout=args.encoder_dropout, 
                                        share_weights=args.encoder_share_weights)
                self.gat = self.gat.to(torch.device(self.device))

                self.video_encoder = VideoEncoder(input_size=args.hidden_dim, hidden_size=args.hidden_dim//4, 
                                                num_layers=2, output_size=args.condition_propagation_dim, max_len=1512)
                self.video_encoder = self.video_encoder.to(torch.device(self.device))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def temporal_encoding(self, tgt, t):

        seq_len, batch_size, _ = tgt.size()
        freq = torch.exp(torch.arange(0, tgt.size(-1), 2) * (-math.log(10000.0) / tgt.size(-1)))
        t = t * freq.unsqueeze(0).unsqueeze(1)  # Broadcasting to match
        temporal_enc = torch.zeros_like(tgt)
        temporal_enc[:, :, 0::2] = torch.sin(t)
        temporal_enc[:, :, 1::2] = torch.cos(t)

        return temporal_enc+ tgt


    def forward(self, src, tgt, mask, tgt_mask, detections, target_nodes, tgt_key_padding_mask, query_embed, pos_embed, tgt_pos_embed, mode='train'):

        graph_output, importance_loss, active_idx = None, None, None

        if self.args.kg_attn == True or self.args.kg_init == True:
            relations = detections[1]
            detections = detections[0]
            
            if self.args.graph_merging:
                self.graph = merge_graphs(self.kg, relations, detections)
                self.graph.getGlobalAdjacencyMat()
            else: 
                self.graph = self.kg
                self.graph.getGlobalAdjacencyMat()

            conditioning_input = None
            if self.args.condition_propagation: 
                conditioning_input = self.video_encoder(src.transpose(0, 1))
        
            if self.args.use_gsnn:
                importance_loss, context_vectors , active_idx= get_context_vectors(self.args, self.gsnn_net, self.graph, detections, target_nodes,
                                                                            conditioning_input=conditioning_input, mode=mode)
                graph_output = context_vectors

            else:
                conditioning_input = self.video_encoder(src.transpose(0, 1))

                node_representations = get_node_representations(self.args, self.graph, self.gat, device=self.device,
                                                                    conditioning_input=conditioning_input)
                graph_output = node_representations    

            if self.args.kg_init:
                sum_pooled_tensor = torch.stack([torch.sum(output, dim=0) for output in graph_output])
                tgt = sum_pooled_tensor.unsqueeze(0).expand(tgt.size(0), -1, -1)

            memory = self.encoder(src, graph_output, src_key_padding_mask=mask, pos=pos_embed)
            intermed_tgt = []
            for t in range(self.T, 0, -1):
                tgt = self.temporal_encoding(tgt, t)
                tgt = self.decoder(tgt, memory, graph_output,tgt_mask=tgt_mask, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_key_padding_mask,
                            pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed)
                intermed_tgt.append(tgt)
            # intermed_tgt = torch.stack(intermed_tgt)
        else:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
            intermed_tgt = []
            for t in range(self.T, 0, -1):
                tgt = self.temporal_encoding(tgt, t)
                tgt = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_key_padding_mask,
                            pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed)
                intermed_tgt.append(tgt)
            # intermed_tgt = torch.stack(intermed_tgt)

        return memory, intermed_tgt, [importance_loss, active_idx]
        # return memory, tgt, importance_loss


class Transformer(nn.Module):

    def __init__(self, args, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        self.args = args
        self.d_head = d_head = d_model // nhead

        if args.kg_attn == True or args.kg_init == True: 
            self.knowledge_weighting_encoder_model = KnowledgeWeightingModel(args)
            self.knowledge_weighting_decoder_model = KnowledgeWeightingModel(args)

            encoder_layer = TransformerEncoderLayer(args, d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,
                                                    self.knowledge_weighting_encoder_model)

            decoder_layer = TransformerDecoderLayer(args, d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,
                                                    self.knowledge_weighting_decoder_model)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            if args.mamba:
                self.encoder = MixerModel( d_model =d_model, n_layer= num_encoder_layers)
            else:
                self.encoder = TransformerEncoder(args, encoder_layer, num_encoder_layers, encoder_norm)

            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(args, decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        else:
            encoder_layer = TransformerEncoderLayer(args,d_model, nhead, dim_feedforward,dropout, activation, normalize_before)
            # encoder_layer = MambaBlock(args, d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
            # encoder_layer = MambaBlock2(args, d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            if args.mamba:
                self.encoder = MixerModel( d_model =d_model, n_layer= num_encoder_layers)
            else:
                self.encoder = TransformerEncoder(args, encoder_layer, num_encoder_layers, encoder_norm)

            decoder_layer = TransformerDecoderLayer(args,d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(args, decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.device = torch.device('cuda')

        if args.kg_attn == True or args.kg_init == True:
            if self.args.graph_merging:
                args.vocab_size = 500   #Setting a high vocab size becauce number of nodes vary for in merged graphs.

            self.kg = Graph()
            self.kg = pickle.load(open('./datasets/graph_kitchen.pkl', 'rb'))
            # self.graph.getGlobalAdjacencyMat()

            if args.use_gsnn:
                self.gsnn_net = GSNN(args)
                self.gsnn_net = self.gsnn_net.to(torch.device(self.device))

                if args.condition_propagation:
                    self.video_encoder = VideoEncoder(input_size=args.hidden_dim, hidden_size=args.hidden_dim//4, 
                                                num_layers=2, output_size=args.condition_propagation_dim, max_len=1512)
                    self.video_encoder = self.video_encoder.to(torch.device(self.device))
            
            else:
                self.gat = ModifiedGATv2(args, in_features=args.state_dim*2, n_hidden=args.state_dim, 
                                        n_heads=args.state_dim, dropout=args.encoder_dropout, 
                                        share_weights=args.encoder_share_weights)
                self.gat = self.gat.to(torch.device(self.device))

                self.video_encoder = VideoEncoder(input_size=args.hidden_dim, hidden_size=args.hidden_dim//4, 
                                                num_layers=2, output_size=args.condition_propagation_dim, max_len=1512)
                self.video_encoder = self.video_encoder.to(torch.device(self.device))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, mask, tgt_mask, detections, target_nodes, tgt_key_padding_mask, query_embed, pos_embed, tgt_pos_embed, mode='train'):

        graph_output, importance_loss, active_idx = None, None, None

        if self.args.kg_attn == True or self.args.kg_init == True:
            relations = detections[1]
            detections = detections[0]
            pdb.set_trace()

            if self.args.graph_merging:
                self.graph = merge_graphs(self.kg, relations, detections)
                self.graph.getGlobalAdjacencyMat()
            else:
                self.graph = self.kg
                self.graph.getGlobalAdjacencyMat()
                
            conditioning_input = None
            if self.args.condition_propagation: 
                conditioning_input = self.video_encoder(src.transpose(0, 1))
        
            if self.args.use_gsnn:
                importance_loss, context_vectors, active_idx = get_context_vectors(self.args, self.gsnn_net, self.graph, detections, target_nodes,
                                                                            conditioning_input=conditioning_input, mode=mode)
                graph_output = context_vectors

            else:
                conditioning_input = self.video_encoder(src.transpose(0, 1))

                node_representations = get_node_representations(self.args, self.graph, self.gat, device=self.device,
                                                                    conditioning_input=conditioning_input)
                graph_output = node_representations 

            if self.args.kg_init:
                sum_pooled_tensor = torch.stack([torch.sum(output, dim=0) for output in graph_output])
                tgt = sum_pooled_tensor.unsqueeze(0).expand(tgt.size(0), -1, -1)   

            memory = self.encoder(src, graph_output, src_key_padding_mask=mask, pos=pos_embed)
            hs = self.decoder(tgt, memory, graph_output, tgt_mask=tgt_mask, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_key_padding_mask,
                            pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed)
        else:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
            hs = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=mask, tgt_key_padding_mask=tgt_key_padding_mask,
                           pos=pos_embed, query_pos=query_embed, tgt_pos=tgt_pos_embed)


        return memory, hs, [importance_loss, active_idx]

class TransformerEncoder(nn.Module) :

    def __init__(self, args, encoder_layer, num_layers, norm=None) :
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, graph_output = None,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, graph_output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, args, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, graph_output=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, graph_output, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, tgt_pos=tgt_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, args, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, knowledge_model=None):
        super().__init__()

        self.args = args

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
  
        if args.kg_attn == True:
            self.self_attn = MultiheadAttention(args, d_model, nhead, dropout=dropout, knowledge_model=knowledge_model)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     graph_output=None,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        q = k = v = self.with_pos_embed(src, pos)

        if self.args.kg_attn == True:
            src2 = self.self_attn(q, k, value=v, graph_output=graph_output, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        else:
            src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, graph_output,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = v = self.with_pos_embed(src2, pos)
        if self.args.kg_attn == True:
            src2 = self.self_attn(q, k, value=v, graph_output=graph_output, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        else:
            src2 = self.self_attn(q, k, value=v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, graph_output = None,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, graph_output, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, graph_output, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, args, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, knowledge_model=None):
        super().__init__()

        self.args = args

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        if args.kg_attn == True:
            self.self_attn = MultiheadAttention(args, d_model, nhead, dropout=dropout, knowledge_model=knowledge_model)
            self.multihead_attn = MultiheadAttention(args, d_model, nhead, dropout=dropout, knowledge_model=knowledge_model)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, graph_output,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] =None):

        q = k = v = self.with_pos_embed(tgt, query_pos)
            
        if self.args.kg_attn == True:
            tgt2 = self.self_attn(q, k, value=v, graph_output=graph_output, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
            
        if self.args.kg_attn == True:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=self.with_pos_embed(memory, pos),
                                   graph_output=graph_output,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=self.with_pos_embed(memory, pos),
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, graph_output,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = v = self.with_pos_embed(tgt2, query_pos)
        
        if self.args.kg_attn == True:
            tgt2 = self.self_attn(q, k, value=v, graph_output=graph_output,
                                attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)[0]
        else:
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        
        if self.args.kg_attn == True:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, graph_output=graph_output, 
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        else:
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory,  
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, graph_output=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, graph_output, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, graph_output, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")





def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=True,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, d_state= 4,layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    # mixer_cls = partial(Mamba,layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block



class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # pdb.set_trace()


        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )



    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states,graph_output = None, src_key_padding_mask = None,pos= None , inference_params=None):
        hidden_states = rearrange(hidden_states, 't b c -> b t c')
        residual = None
        # pdb.set_trace()
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        hidden_states = rearrange(hidden_states, 'b t c -> t b c')
        return hidden_states

