import numpy as np

import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable






class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.net = nn.Sequential(
            nn.Linear(self.configs.input_length, self.configs.input_length*4),
            nn.LeakyReLU(),
            nn.Linear(self.configs.input_length*4, self.configs.input_length),
        )
    def forward(self, x):
        return self.net(x)
class ConvAttention(nn.Module):
    def __init__(self,configs):
        super(ConvAttention, self).__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel=self.num_hidden[-1], out_channel=3*self.num_hidden[-1], kernel_size=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel=self.num_hidden[-1], out_channel=self.num_hidden[-1], kernel_size=5,padding=2)
        )
    def forward(self, x ,enc_out= None,dec=False):
        b,c,h,w,l = x.shape
        qkv_setlist = []
        Vout_list = []
        for i in l:
            qkv_setlist.append(self.conv1(x[...,i]))
        qkv_set = torch.stack(qkv_setlist,dim=-1)
        if dec:
            Q,K,V = torch.split(qkv_set,self.num_hidden[-1],dim=1)
        else:
            Q,K,_ = torch.split(qkv_set,self.num_hidden[-1],dim=1)
            V = enc_out

        for i in l:
            Qi = rearrange([Q[...,i]]*l+K, 'b n h w l -> (b l) n h w')
            tmp = rearrange(self.conv2(Qi),'(b l) n h w -> b n h w l',l=l)
            tmp = F.softmax(tmp, dim=4) #(b, n, h, w, l)
            tmp = np.multiply(tmp, torch.stack([V[i]]*l, dim=-1))
            Vout_list.append(torch.sum(tmp,dim=4)) #(b,n,h,w)
        Vout = torch.stack(Vout_list, dim=-1 )
        return Vout                            #(b,n,h,w,l)

class PositionalEncoding(nn.Module):

    def __init__(self, configs):
        super(PositionalEncoding, self).__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table())

    def _get_sinusoid_encoding_table(self):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):

            return_list = [torch.ones((self.configs.batch_size,
                                       self.configs.img_width,
                                       self.configs.img_width)).to(self.configs.device)*(position / np.power(10000, 2 * (hid_j // 2) / self.num_hidden[-1])) for hid_j in range(self.num_hidden[-1])]
            return torch.stack(return_list, dim=1)
        sinusoid_table = [get_position_angle_vec(pos_i) for pos_i in range(self.configs.input_length)]
        sinusoid_table[0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.stack(sinusoid_table, dim=-1)

    def forward(self, x):
        '''

        :param x: (b, channel, h, w, seqlen)
        :return:
        '''
        return x + self.pos_table.clone().detach()

class Encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm([self.num_hidden[-1],self.configs.img_width,self.configs.img_width],
                                 ConvAttention(self.configs))),
                Residual(PreNorm([self.num_hidden[-1],self.configs.img_width,self.configs.img_width],
                                 FeedForward(self.configs)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm([self.num_hidden[-1], self.configs.img_width, self.configs.img_width],
                                 ConvAttention(self.configs))),
                Residual(PreNorm([self.num_hidden[-1], self.configs.img_width, self.configs.img_width],
                                 FeedForward(self.configs)))
            ]))
    def forward(self, x, enc_out, mask=None):
        for attn, ff in (self.layers):
            x = attn(x,enc_out=enc_out,dec=True)
            x = ff(x)
        return x

class feature_generator(nn.Module):
    def __init__(self,configs):
        super(feature_generator, self).__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.num_hidden[0],
                               kernel_size=self.configs.filter_size,
                               stride=1,
                               padding=(self.configs.filter_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels=self.num_hidden[0],
                               out_channels=self.num_hidden[1],
                               kernel_size=self.configs.filter_size,
                               stride=1,
                               padding=(self.configs.filter_size-1)//2)
        self.conv3 = nn.Conv2d(in_channels=self.num_hidden[1],
                               out_channels=self.num_hidden[2],
                               kernel_size=self.configs.filter_size,
                               stride=1,
                               padding=(self.configs.filter_size-1)//2)
        self.conv4 = nn.Conv2d(in_channels=self.num_hidden[2],
                               out_channels=self.num_hidden[3],
                               kernel_size=self.configs.filter_size,
                               stride=1,
                               padding=(self.configs.filter_size-1)//2)
        self.bn1 = nn.BatchNorm2d(self.num_hidden[0])
        self.bn2 = nn.BatchNorm2d(self.num_hidden[1])
        self.bn3 = nn.BatchNorm2d(self.num_hidden[2])
        self.bn4 = nn.BatchNorm2d(self.num_hidden[3])
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn3(self.conv3(out)), negative_slope=0.01, inplace=False)
        out = F.leaky_relu(self.bn4(self.conv4(out)), negative_slope=0.01, inplace=False)
        return out

# def feature_embedding(img, configs):
#     generator = feature_generator(configs).to(configs.device)
#     gen_img = []
#     for i in range(img.shape[-1]):
#         gen_img.append(generator(img[:,:,:,:,i]))
#     gen_img = torch.stack(gen_img, dim=-1)
#     return gen_img

class Transformer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.num_hidden = [int(x) for x in self.configs.num_hidden.split(',')]
        self.pos_embedding = PositionalEncoding(self.configs)
        self.Encoder = Encoder(dim, depth, heads, mlp_dim, dropout)
        self.Decoder = Decoder(dim,depth, heads, mlp_dim, dropout)
        self.back_to_pixel = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1],1,kernel_size=1)
        )
    def forward(self, frames, mask = None):
        b,n,h,w,l = frames.shape
        out_list=[]
        feature_map = self.feature_embedding(img=frames,configs=self.configs)
        enc_in = self.pos_embedding(feature_map)
        enc_out = self.Encoder(enc_in)
        # TODO decide which one is the query: feature_map or frames
        dec_out = self.Decoder(enc_in,enc_out)
        for i in l:
            out_list.append(self.back_to_pixel(dec_out[...,i]))
        x = torch.stack(out_list,dim=-1)
        return x

    def feature_embedding(self,img, configs):
        generator = feature_generator(configs).to(configs.device)
        gen_img = []
        for i in range(img.shape[-1]):
            gen_img.append(generator(img[:, :, :, :, i]))
        gen_img = torch.stack(gen_img, dim=-1)
        return gen_img


# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dropout = 0.):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim ** -0.5
#
#         self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
#         self.to_out = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x,kv = None,first = True ,mask = None):  # x shape(1,65,1024)
#         b, n, _, h = *x.shape, self.heads
#         if first:
#             qkv = self.to_qkv(x)  #(1,65,1024*3)
#             q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # (1,8,65,128)
#         else :
#             q = rearrange(x,'b n (h d) -> b h n d',h = h)
#             k = rearrange(kv,'b n (h d) -> b h n d',h = h)
#             v = rearrange(kv,'b n (h d) -> b h n d',h = h)
#         dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#
#         if mask is not None:
#             mask = F.pad(mask.flatten(1), (1, 0), value = True)
#             assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
#             mask = mask[:, None, :] * mask[:, :, None]
#             dots.masked_fill_(~mask, float('-inf'))
#             del mask
#
#         attn = dots.softmax(dim=-1)
#
#         out = torch.einsum('bhij,bhjd->bhid', attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out =  self.to_out(out)
#         return out



