import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
from torch.nn.modules.utils import _pair
from mmcv.ops import modulated_deform_conv2d
from lkconv import ReparamLargeKernelConv
import torch
import torch.nn as nn
from functools import partial
import vits


class moonlit(nn.Module):
    def __init__(self, opt):
        super(moonlit, self).__init__()
        self.UMRC = torch.nn.DataParallel(UMRC(opt))
        self.LDRN = torch.nn.DataParallel(LDRN(opt))

    def forward(self, x_query, x_key, opt):
        key, query, labels, inter = self.UMRC(x_query, x_key, opt)
        restored = self.LDRN(x_query, inter)
        return restored, query, labels


class ResBlock(nn.Module):
    def __init__(self, in_keyt, out_keyt, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_keyt, out_keyt, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_keyt),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_keyt, out_keyt, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_keyt),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_keyt, out_keyt, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_keyt)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()
        self.UMRC_pre = ResBlock(in_keyt=3, out_keyt=64, stride=1)
        self.UMRC = nn.Sequential(
            ResBlock(in_keyt=64, out_keyt=128, stride=2),
            ResBlock(in_keyt=128, out_keyt=256, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        inter = self.UMRC_pre(x)
        key = self.UMRC(inter).squeeze(-1).squeeze(-1)
        out = self.mlp(key)
        return key, out, inter


class UMRC(nn.Module):
    def __init__(self, opt):
        super(UMRC, self).__init__()
        if opt.net_name == 'resnet':
            self.UMRC = MoCo(base_encoder=ResEncoder, dim=256, K=opt.batch_size * 256)
        elif opt.net_name == 'vit':
            self.UMRC = MoCo(base_encoder=vits.vit_base, dim=256, K=opt.batch_size * 256)

    def forward(self, x_query, x_key, opt):
        key, query, labels, inter = self.UMRC(x_query, x_key, opt)
        return key, query, labels, inter


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class DCN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True):
        super(DCN_layer, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels * 2,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_keyt, inter):
        keyt_degradation = torch.cat([input_keyt, inter], dim=1)
        out = self.conv_offset_mask(keyt_degradation)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(input_keyt.contiguous(), offset, mask, self.weight, self.bias, self.stride,
                                       self.padding, self.dilation, self.groups, self.deformable_groups)


class SFT_layer(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)
        return x * gamma + beta

class miniLDB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(miniLDB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.body = nn.Sequential(*[DCN_layer(self.in_channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=False),
                                    SFT_layer(self.in_channels, self.out_channels)])

    def forward(self, x, inter):
        return x + self.body[0](x, inter) + self.body[1](x, inter)
    
class LDB(nn.Module):
    def __init__(self, conv, n_keyt, kernel_size):
        super(LDB, self).__init__()
        self.body = nn.Sequential(*[miniLDB(n_keyt, n_keyt, kernel_size),
                                    nn.LeakyReLU(0.1, True),
                                    ReparamLargeKernelConv(n_keyt, n_keyt, kernel_size=31, stride=1, groups=8, small_kernel=5),
                                    nn.LeakyReLU(0.1, True),
                                    miniLDB(n_keyt, n_keyt, kernel_size),
                                    nn.LeakyReLU(0.1, True),
                                    conv(n_keyt, n_keyt, kernel_size)])

    def forward(self, x, inter):
        out = self.body[0](x, inter)
        out = self.body[1](out)
        out = self.body[2](out)
        out = self.body[3](out)
        out = self.body[4](out, inter)
        out = self.body[5](out)
        out = self.body[6](out) + x
        return out


class LDG(nn.Module):
    def __init__(self, conv, n_keyt, kernel_size):
        super(LDG, self).__init__()
        self.body = nn.Sequential(*[LDB(conv, n_keyt, kernel_size),
                                    LDB(conv, n_keyt, kernel_size),
                                    LDB(conv, n_keyt, kernel_size),
                                    LDB(conv, n_keyt, kernel_size),
                                    LDB(conv, n_keyt, kernel_size),
                                    ReparamLargeKernelConv(n_keyt, n_keyt, kernel_size=31, stride=1, groups=8, small_kernel=5)])

    def forward(self, x, inter):
        query = x
        query = self.body[0](query, inter)
        query = self.body[1](query, inter)
        query = self.body[2](query, inter)
        query = self.body[3](query, inter)
        query = self.body[4](query, inter)
        query = self.body[-1](query)
        query = query + x
        return query


class LDRN(nn.Module):
    def __init__(self, opt, conv=default_conv):
        super(LDRN, self).__init__()
        self.body = nn.Sequential(*[ReparamLargeKernelConv(3, 64, kernel_size=31, stride=1, groups=1, small_kernel=5),
                                    LDG(default_conv, 64, 3),
                                    LDG(default_conv, 64, 3),
                                    LDG(default_conv, 64, 3),
                                    LDG(default_conv, 64, 3),
                                    LDG(default_conv, 64, 3), conv(64, 64, 3),
                                    ReparamLargeKernelConv(64, 3, kernel_size=31, stride=1, groups=1, small_kernel=5)])

    def forward(self, x, query):
        x = self.body[0](x)
        inter = x
        inter = self.body[1](inter, query)
        inter = self.body[2](inter, query)
        inter = self.body[3](inter, query)
        inter = self.body[4](inter, query)
        inter = self.body[5](inter, query)
        inter = self.body[6](inter)
        inter = inter + x
        x = self.body[7](inter)
        return x


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, K=3*256, m=0.999, T=0.07):
        """
        dim: keyture dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        self.K = self.K * batch_size
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def _build_projector_and_predictor_mlps(self, dim=256, mlp_dim=4096):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

    def forward(self, im_q, im_k, opt):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            query, targets
        """
        # compute query keytures
        embedding, q, inter = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key keytures
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            _, k, _ = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute query
        # Einstein sum is more intuitive
        # positive query: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative query: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # query: Nx(1+K)
        query = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        query /= self.T

        # labels: positive key indicators
        labels = torch.zeros(query.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return embedding, query, labels, inter

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output