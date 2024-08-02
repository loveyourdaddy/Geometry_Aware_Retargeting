# from collections import OrderedDict
# import numpy as np
import torch
import torch.nn as nn
import math
# from torch import asin, atan2
# from torch.utils.data import Dataset
# from os import listdir, makedirs
# from os.path import exists, join



class FeedForward(nn.Module):  # FFN
    def __init__(self, args, input_dim, output_dim):
        super().__init__()
        hid_dim = 256
        dropout = 1 - args.kp
        self.net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hid_dim),
            nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            nn.Linear(in_features=hid_dim, out_features=hid_dim),
            nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            nn.Linear(in_features=hid_dim, out_features=output_dim),
        )

    def forward(self, input):
        output = self.net(input)
        return output

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first, num_layers):
        super().__init__()
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            num_layers=num_layers,
        )
        self.i2h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.i2h(output)
        return output, hidden
        
import torch.nn.utils.rnn as rnn_utils
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first, num_layers):
        super().__init__()
        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            num_layers=num_layers,
        )
        self.i2h = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input, hidden, length_list):
        # input 
        # hidden (hidden, cell)
        packed = rnn_utils.pack_padded_sequence(input, length_list, batch_first=True, enforce_sorted=False) # 
        output, hidden = self.rnn(packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.i2h(output)
        return output, hidden
    
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=22):
        self.odd_dim = False
        if dim % 2 != 0:
            dim += 1
            self.odd_dim = True
            # raise ValueError(
            #     "Cannot use sin/cos positional encoding with "
            #     "odd dim (got dim={:d})".format(dim)
            # )
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, n, self.dim)``
        """
        if self.odd_dim:
            emb = torch.cat([emb, emb.new_zeros(emb.size(0), emb.size(1), 1)], dim=2)

        emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[:, 0 : emb.size(1), :]
        emb = self.dropout(emb) 

        if self.odd_dim:
            emb = emb[..., :-1]
        return emb

""" Implementation of InterGen """
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, args, spat_num_token, max_temp_num_token, char_dim):
        super(SpatioTemporalTransformer, self).__init__()
        self.args = args
        kp = args.kp
        dropout = 1 - kp 
        self.spat_num_token = spat_num_token
        
        # input dim 
        root_dim = 3
        input_dim = args.motion_dim + 2*char_dim
        self.input_dim = input_dim
        self.char_dim = char_dim
        # emb dim
        rot_token_channels = 32 # rot emb
        rest_token_channels = 16
        root_dim = 3 
        embed_channels = 16 # final emb dim 
        self.rot_token_channels = rot_token_channels
        
        # root net 
        self.root_net_a = nn.Linear(root_dim, input_dim)
        self.root_net_b = nn.Linear(root_dim, input_dim)
        
        # Transformer 
        self.spat_pos_encoder = PositionalEncoding(dropout, input_dim, max_len=spat_num_token)
        if args.temporal_attn==False:
            if args.weight_sharing==False:
                # base 
                self.spat_encoder = TwinTransformer(args, input_dim, rot_token_channels, dropout)
            else:
                # sharing 
                self.spat_encoder = SharingTransformer(args, input_dim, rot_token_channels, dropout)
        elif args.temporal_attn:
            self.temp_pos_encoder = PositionalEncoding(dropout, input_dim, max_len=max_temp_num_token) # self.args.window_size
            if args.weight_sharing==False:
                # temporal 
                self.spat_encoder = TwinTransformer(args, input_dim, rot_token_channels, dropout)
            elif args.weight_sharing:
                # sharing temporal
                self.spat_encoder = SharingTemporalTransformer(args, input_dim, rot_token_channels, dropout)
        else: 
            raise NotImplementedError
        
        # proj
        total_embed_channels = 1*(rot_token_channels) #  + root_dim + 2*rest_token_channels
        # rot 
        self.embed_linear_a = nn.Linear(spat_num_token*total_embed_channels, self.args.num_joint*embed_channels)
        self.embed_linear_b = nn.Linear(spat_num_token*total_embed_channels, self.args.num_joint*embed_channels)
        self.embed_acti = nn.ReLU()
        self.embed_drop = nn.Dropout(dropout)
        self.delta_linear_a = nn.Linear(self.args.num_joint*embed_channels, self.args.num_joint*args.rot_dim)
        self.delta_linear_b = nn.Linear(self.args.num_joint*embed_channels, self.args.num_joint*args.rot_dim)
        # root_p
        self.root_embed_linear_a = nn.Linear(spat_num_token*total_embed_channels, embed_channels)
        self.root_embed_linear_b = nn.Linear(spat_num_token*total_embed_channels, embed_channels)
        self.root_embed_acti = nn.ReLU()
        self.root_embed_drop = nn.Dropout(dropout)
        self.root_delta_linear_a = nn.Linear(embed_channels, 3)
        self.root_delta_linear_b = nn.Linear(embed_channels, 3)

    def forward(self, qa_t, qb_t, root_a, root_b, pos_a_t, pos_b_t, sour_restA, sour_restB, targ_restA, targ_restB):
        batch_size, len_frame, _ = qa_t.shape
        num_joint, rot_dim = self.args.num_joint, self.args.rot_dim
        
        # input per joint (not root)
        qa_t_ = qa_t.reshape(batch_size, len_frame, num_joint, rot_dim)
        qb_t_ = qb_t.reshape(batch_size, len_frame, num_joint, rot_dim)
        sour_restA_ = sour_restA.reshape(batch_size, len_frame, num_joint, self.char_dim)
        sour_restB_ = sour_restB.reshape(batch_size, len_frame, num_joint, self.char_dim)
        targ_restA_ = targ_restA.reshape(batch_size, len_frame, num_joint, self.char_dim)
        targ_restB_ = targ_restB.reshape(batch_size, len_frame, num_joint, self.char_dim)
        concat_a_t = torch.cat((qa_t_, pos_a_t, sour_restA_, targ_restA_), dim=-1)
        concat_b_t = torch.cat((qb_t_, pos_b_t, sour_restB_, targ_restB_), dim=-1)
        
        # root net
        root_a_ = self.root_net_a(root_a).unsqueeze(-2)
        root_b_ = self.root_net_b(root_b).unsqueeze(-2)
        # concat 
        concat_a_t = torch.cat((concat_a_t, root_a_), dim=-2)
        concat_b_t = torch.cat((concat_b_t, root_b_), dim=-2)
        
        # reshape (batch*frame, 22, input_dim)
        spat_a_t = concat_a_t.reshape(-1, self.spat_num_token, self.input_dim)
        spat_b_t = concat_b_t.reshape(-1, self.spat_num_token, self.input_dim)
        
        """ Encoder """
        # spatio positional encoding 
        spat_qa_embed = self.spat_pos_encoder(spat_a_t)
        spat_qb_embed = self.spat_pos_encoder(spat_b_t)
        
        # net
        if self.args.temporal_attn == False:
            # base, sharing
            out_a, out_b = \
                self.spat_encoder(spat_qa_embed, spat_qb_embed)
        else:
            # temporal positional encoding 
            temp_a_t = concat_a_t.transpose(1, 2)
            temp_b_t = concat_b_t.transpose(1, 2)
            temp_qa_embed = []
            temp_qb_embed = []
            for i in range(batch_size):
                temp_qa_embed.append(self.temp_pos_encoder(temp_a_t[i]))
                temp_qb_embed.append(self.temp_pos_encoder(temp_b_t[i]))
            temp_qa_embed = torch.stack(temp_qa_embed, dim=0)
            temp_qb_embed = torch.stack(temp_qb_embed, dim=0)
            temp_qa_embed = temp_qa_embed.reshape(-1, len_frame, self.input_dim)
            temp_qb_embed = temp_qb_embed.reshape(-1, len_frame, self.input_dim)
            
            if self.args.weight_sharing==False:
                # temporal
                out_a, out_b = \
                    self.spat_encoder(temp_qa_embed, temp_qb_embed)
            else:
                # sharing temporal 
                out_a, out_b = \
                    self.spat_encoder(spat_qa_embed, spat_qb_embed, temp_qa_embed, temp_qb_embed)
            out_a = out_a.reshape(batch_size, self.spat_num_token, len_frame, self.rot_token_channels).transpose(1, 2)
            out_b = out_b.reshape(batch_size, self.spat_num_token, len_frame, self.rot_token_channels).transpose(1, 2)
            
        """ Decoder  """
        # emb 
        cat_embed_a = out_a.reshape(batch_size, len_frame, -1)
        cat_embed_b = out_b.reshape(batch_size, len_frame, -1)
        # rot 
        embed_a = self.embed_drop(self.embed_acti(self.embed_linear_a(cat_embed_a))) # 23*32 
        embed_b = self.embed_drop(self.embed_acti(self.embed_linear_b(cat_embed_b))) 
        deltaq_a_t = self.delta_linear_a(embed_a)
        deltaq_b_t = self.delta_linear_b(embed_b)
        
        # root_p
        root_embed_a = self.root_embed_drop(self.root_embed_acti(self.root_embed_linear_a(cat_embed_a)))
        root_embed_b = self.root_embed_drop(self.root_embed_acti(self.root_embed_linear_b(cat_embed_b)))
        root_embed_a = root_embed_a.reshape(batch_size, len_frame, -1)
        root_embed_b = root_embed_b.reshape(batch_size, len_frame, -1)
        root_deltaq_a_t = self.root_delta_linear_a(root_embed_a)
        root_deltaq_b_t = self.root_delta_linear_b(root_embed_b)
        
        # concat
        delta_a_t = torch.cat((deltaq_a_t, root_deltaq_a_t), dim=-1)
        delta_b_t = torch.cat((deltaq_b_t, root_deltaq_b_t), dim=-1)
        
        return delta_a_t, delta_b_t

""" Transformer module """

class Transformer(nn.Module):
    def __init__(self, args, input_dim, token_channels, dropout):
        super(Transformer, self).__init__()

        self.token_linear = nn.Linear(input_dim, token_channels)

        # transformer 
        num_layers = 2
        latent_dim = token_channels
        num_heads = args.num_heads

        self.num_layers = num_layers
        self.trans0 = nn.ModuleList()
        for i in range(num_layers):
            self.trans0.append(
                TransformerBlock(latent_dim=latent_dim, 
                                 num_heads=num_heads, 
                                 dropout=dropout,))

    # pose_t: bs, joint, 4
    def forward(self, pose_a_t):
        # emb 
        token_a_q = self.token_linear(pose_a_t)
        # trf 
        for i in range(self.num_layers):
            token_a_q = self.trans0[i](token_a_q)

        return token_a_q
    
class TwinTransformer(nn.Module):
    def __init__(self, args, input_dim, token_channels, dropout):
        super(TwinTransformer, self).__init__()
        self.trans0 = Transformer(args, input_dim, token_channels, dropout)
        self.trans1 = Transformer(args, input_dim, token_channels, dropout)

    # pose_t: bs, joint, 4
    def forward(self, pose_a_t, pose_b_t):
        token_a_q = self.trans0(pose_a_t)
        token_b_q = self.trans1(pose_b_t)
        
        return token_a_q, token_b_q
    
class SharingTransformer(nn.Module):
    def __init__(self, args, input_dim, token_channels, dropout):
        super(SharingTransformer, self).__init__()

        # self.num_joint = num_joint
        self.token_linear = nn.Linear(input_dim, token_channels)
        
        # transformer 
        num_layers = 2
        latent_dim = token_channels
        num_heads = args.num_heads
        # dropout = kp

        self.num_layers = num_layers
        self.trans0 = nn.ModuleList()
        self.trans1 = nn.ModuleList()
        for i in range(num_layers):
            self.trans0.append(
                CrossTransformerBlock(latent_dim=latent_dim, 
                                 num_heads=num_heads, 
                                 dropout=dropout,))
            self.trans1.append(
                CrossTransformerBlock(latent_dim=latent_dim, 
                                 num_heads=num_heads, 
                                 dropout=dropout,))

    # pose_t: bs, joint, 4
    def forward(self, pose_a_t, pose_b_t):
        # emb 
        token_a_q = self.token_linear(pose_a_t)
        token_b_q = self.token_linear(pose_b_t)
        # trf 
        for i in range(self.num_layers):
            embed_a_q = self.trans0[i](token_a_q, token_b_q)
            embed_b_q = self.trans1[i](token_b_q, token_a_q)
            token_a_q = embed_a_q
            token_b_q = embed_b_q

        return embed_a_q, embed_b_q

class SharingTemporalTransformer(nn.Module):
    def __init__(self, args, input_dim, token_channels, dropout):
        super(SharingTemporalTransformer, self).__init__()
        self.spat_token_linear = nn.Linear(input_dim, token_channels)
        self.temp_token_linear = nn.Linear(input_dim, token_channels)
        
        # transformer 
        num_layers = 2
        self.latent_dim = token_channels
        latent_dim = token_channels
        num_heads = args.num_heads
        # dropout = kp

        self.num_layers = num_layers
        self.trans0 = nn.ModuleList()
        self.trans1 = nn.ModuleList()
        for i in range(num_layers):
            self.trans0.append(
                TemporalCrossTransformerBlock(latent_dim=latent_dim, 
                                 num_heads=num_heads, 
                                 dropout=dropout,))
            self.trans1.append(
                TemporalCrossTransformerBlock(latent_dim=latent_dim, 
                                 num_heads=num_heads, 
                                 dropout=dropout,))

    # pose_t: bs, joint, 4
    def forward(self, spat_pose_a_t, spat_pose_b_t, temp_pose_a_t, temp_pose_b_t):
        _, num_token, _ = spat_pose_a_t.shape
        _, frame_size, _ = temp_pose_a_t.shape 
        token_dim = self.latent_dim
        batch_size = temp_pose_a_t.shape[0] // num_token
        
        # emb 
        spat_token_a_q = self.spat_token_linear(spat_pose_a_t)
        spat_token_b_q = self.spat_token_linear(spat_pose_b_t)
        temp_token_a_q = self.temp_token_linear(temp_pose_a_t)
        temp_token_b_q = self.temp_token_linear(temp_pose_b_t)
        # trf 
        for i in range(self.num_layers):
            embed_a_q = self.trans0[i](spat_token_a_q, temp_token_a_q, spat_token_b_q)
            embed_b_q = self.trans1[i](spat_token_b_q, temp_token_b_q, spat_token_a_q)
            spat_token_a_q = embed_a_q
            spat_token_b_q = embed_b_q
            # reshape
            temp_token_a_q = embed_a_q.reshape(batch_size, frame_size, num_token, token_dim)
            temp_token_b_q = embed_b_q.reshape(batch_size, frame_size, num_token, token_dim)
            temp_token_a_q = temp_token_a_q.transpose(1, 2).reshape(-1, frame_size, token_dim)
            temp_token_b_q = temp_token_b_q.transpose(1, 2).reshape(-1, frame_size, token_dim)
            
        return embed_a_q, embed_b_q

    
""" block """
class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim, # 64
                #  ff_size, # 32  # 더 커야 
                 num_heads,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl
        ff_size = 2*latent_dim # hidden dim 

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        out = self.ffn(h1, emb)
        out = out + h1
        return out
    
class CrossTransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim, # 64
                #  ff_size, # 32  # 더 커야 
                 num_heads,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl
        ff_size = 2*latent_dim # hidden dim 

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out

class TemporalCrossTransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim, # 64
                #  ff_size, # 32  # 더 커야 
                 num_heads,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl
        ff_size = 2*latent_dim # hidden dim 

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.te_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, spat_x, temp_x, spat_y, emb=None, key_padding_mask=None): # temp_y, 
        _, num_token, token_dim = spat_x.shape
        _, frame_size, _ = temp_x.shape 
        batch_size = temp_x.shape[0] // num_token
        
        # spatial
        spat_h1 = self.sa_block(spat_x, emb, key_padding_mask)
        spat_h1 = spat_h1 + spat_x
        # temporal
        temp_h1 = self.te_block(temp_x, emb, key_padding_mask)
        temp_h1 = temp_h1 + temp_x
        temp_h1 = temp_h1.reshape(batch_size, num_token, frame_size, token_dim)
        temp_h1 = temp_h1.transpose(1, 2).reshape(-1, num_token, token_dim)
        h1_sum = spat_h1 + temp_h1 
        
        # cross
        h2 = self.ca_block(h1_sum, spat_y, emb, key_padding_mask)
        h2 = h2 + h1_sum
        out = self.ffn(h2, emb)
        out = out + h2
        return out

""" attention operator"""
class VanillaSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.attention = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout, batch_first=True,
                                               add_zero_attn=True)
        self.norm = AdaLN(latent_dim, embed_dim)

    def forward(self, x, emb, key_padding_mask=None):
        """
        x: B, T, D
        """
        y = self.attention(x, x, x,
                           attn_mask=None,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        y_norm = self.norm(y) # , emb
        return y_norm

class VanillaCrossAttention(nn.Module):

    def __init__(self, latent_dim, xf_latent_dim, num_head, dropout, embed_dim=None):
        super().__init__()
        self.num_head = num_head
        self.norm = AdaLN(latent_dim, embed_dim)
        self.xf_norm = AdaLN(xf_latent_dim, embed_dim)
        self.attention = nn.MultiheadAttention(latent_dim, num_head, kdim=xf_latent_dim, vdim=xf_latent_dim,
                                               dropout=dropout, batch_first=True, add_zero_attn=True)

    def forward(self, x, xf, emb, key_padding_mask=None):
        """
        x: B, T, D
        xf: B, N, L
        """
        # xf_norm = self.xf_norm(xf, emb)
        y = self.attention(x, xf, xf,
                           attn_mask=None,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        y_norm = self.norm(y) # , emb
        return y_norm


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout, embed_dim=None):
        super().__init__()
        self.norm = AdaLN(latent_dim, embed_dim)
        # self.linear1 = nn.Linear(latent_dim, latent_dim, bias=True)
        self.linear1 = nn.Linear(latent_dim, ffn_dim, bias=True)
        self.linear2 = nn.Linear(ffn_dim, latent_dim, bias=True) #zero_module(nn.Linear(ffn_dim, latent_dim, bias=True))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, emb=None):
        # if emb is not None:
        #     x_norm = self.norm(x, emb)
        # else:
        #     x_norm = x
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # y = self.linear1(x)
        return y

class AdaLN(nn.Module):

    def __init__(self, latent_dim, embed_dim=None):
        super().__init__()
        # if embed_dim is None:
        #     embed_dim = latent_dim
        # self.emb_layers = nn.Sequential(
        #     # nn.Linear(embed_dim, latent_dim, bias=True),
        #     nn.SiLU(),
        #     nn.Linear(embed_dim, 2 * latent_dim, bias=True), # zero_module(nn.Linear(embed_dim, 2 * latent_dim, bias=True)),
        # )
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, h): # , emb
        """
        h: B, T, D
        emb: B, D
        """
        # # B, 1, 2D
        # emb_out = self.emb_layers(emb)
        # # scale: B, 1, D / shift: B, 1, D
        # scale, shift = torch.chunk(emb_out, 2, dim=-1)
        h = self.norm(h) # * (1 + scale[:, None]) + shift[:, None]
        return h
