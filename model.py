from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import functools
import torch.nn as nn
from torch.nn import init
import torch.functional as F
from torch.autograd import Variable
print('ok')
def weights_init_normal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weight(net,init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_dropout, use_bias)
    
    def build_conv_block(self,dim,use_dropout,use_bias):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim,dim,kernel_size=3,padding=0,bias=use_bias),
                        nn.InstanceNorm2d(dim),
                        nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.ReflectionPad2d(1)] 
        conv_block += [nn.Conv2d(dim,dim,kernel_size=3,padding=0,bias=use_bias),
                        nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self,x):
        out = x + self.conv_block(x)
        return out

class G(nn.Module):
    def __init__(self,dim=64,device_ids=[]):
        super(G,self).__init__()
        self.device_ids = device_ids
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, dim, kernel_size=7, padding=0,bias=False),
                 nn.InstanceNorm2d(dim),
                 nn.ReLU(True)]
        for i in range(2):
            mult = 2 ** i
            model += [nn.Conv2d(dim * mult, dim * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(dim * mult * 2),
                      nn.ReLU(True)]
        for i in range(9):
            model += [ResnetBlock(dim*4,use_dropout=False,use_bias=False)]
        for i in range(2):
            mult = 2**(2 - i)
            model += [nn.ConvTranspose2d(dim * mult, int(dim * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      nn.InstanceNorm2d(int(dim * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(dim,3,kernel_size=7,padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        use_gpu = len(self.device_ids) > 0
        if (use_gpu):
            assert (torch.cuda.is_available())
        if len(self.device_ids)and isinstance(input.data, torch.cuda.FloatTensor):
            print('Train on GPU...')
            return nn.parallel.data_parallel(self.model, input, self.device_ids)
        else:
            print('Train on CPU...')
            return self.model(input)

class D(nn.Module):
    def __init__(self,dim=64,device_ids=[]):
        super(D,self).__init__()
        self.device_ids = device_ids
        model = [nn.Conv2d(3,dim,kernel_size=4,stride=2,padding=1),
                 nn.LeakyReLU(0.2,True)]
        model += [nn.Conv2d(dim,dim*2,kernel_size=4,stride=2,padding=1,bias=False),
                  nn.InstanceNorm2d(dim*2),
                  nn.LeakyReLU(0.2,True)]
        model += [nn.Conv2d(dim*2, dim*4, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(dim*4),
                  nn.LeakyReLU(0.2,True)]
        model += [nn.Conv2d(dim*4, dim*8, kernel_size=4, stride=1, padding=1, bias=False),
                  nn.InstanceNorm2d(dim*8),
                  nn.LeakyReLU(0.2,True)]
        model += [nn.Conv2d(dim*8,1,kernel_size=4,stride=1,padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        use_gpu = len(self.device_ids) > 0
        if (use_gpu):
            assert (torch.cuda.is_available())
        if len(self.device_ids)and isinstance(input.data, torch.cuda.FloatTensor):
            print('Train on GPU...')
            return nn.parallel.data_parallel(self.model, input, self.device_ids)
        else:
            print('Train on CPU...')
            return self.model(input)

print ('kkk')








# class te(nn.Module):
#     def __init__(self):
#         super(te,self).__init__()
#         norm_layer=nn.InstanceNorm2d
#         kw = 4
#         padw = 1
#         input_nc=3
#         n_layers=3
#         ndf=64
#         use_bias = False
#         sequence = [
#             nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2**n, 8)
#             sequence += [
#                 nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                           kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2**n_layers, 8)
#         sequence += [
#             nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                       kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
#
#         self.model1 = nn.Sequential(*sequence)
#     def forward(self,x):
#         return self.model1(x)
