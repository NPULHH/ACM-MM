# -----------------------------------------------------------
# Dual Semantic Relations Attention Network (DSRAN) implementation based on
# "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives"
# "Learning Dual Semantic Relations with Graph Attention for Image-Text Matching"
# Keyu Wen, Xiaodong Gu, and Qingrong Cheng
# IEEE Transactions on Circuits and Systems for Video Technology, 2020
# Writen by Keyu Wen & Linyang Li, 2020
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import copy
from resnet import resnet152
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
import time
from GAT import GATLayer
from GAT_1 import IGSAN
from relative_embedding import BoxRelationalEmbedding


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B*N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x

class RcnnEncoder(nn.Module):
    def __init__(self, opt):
        super(RcnnEncoder, self).__init__()
        self.embed_size = opt.embed_size
        self.fc_image = nn.Linear(opt.img_dim, self.embed_size)
        self.mlp = MLP(opt.img_dim, opt.embed_size // 2, opt.embed_size, 2)

    def forward(self, images):  # (b, 100, 2048) (b,100,1601+6)
        img_f = self.fc_image(images)
        img_f = self.mlp(images) + img_f
        img_f = l2norm(img_f)
        # img_pe = self.fc_pos(img_pos)
        # img_embs = img_f + img_pe
        return img_f # (b,100,768)

class GridEncoder(nn.Module):
    def __init__(self, opt):
        super(GridEncoder, self).__init__()
        self.embed_size = opt.embed_size
        self.fc_grid = nn.Linear(opt.img_dim, self.embed_size)
        self.init_weights()
        
    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_grid.in_features +
                                  self.fc_grid.out_features)
        self.fc_grid.weight.data.uniform_(-r, r)
        self.fc_grid.bias.data.fill_(0)


    def forward(self, img_grid):  # (b, 100, 2048) (b,100,1601+6)
        img_g = self.fc_grid(img_grid)
        img_g = l2norm(img_g)
        return img_g





class TextEncoder(nn.Module):
    def __init__(self, opt):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(opt.bert_path)
        if not opt.ft_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print('text-encoder-bert no grad')
        else:
            print('text-encoder-bert fine-tuning !')
        self.embed_size = opt.embed_size
        self.fc = nn.Sequential(nn.Linear(opt.bert_size, opt.embed_size), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, captions):
        all_encoders, pooled = self.bert(captions)
        out = all_encoders[-1]
        out = self.fc(out)
        return out


class GATopt(object):
    def __init__(self, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = 8
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2


class GAT(nn.Module):
    def __init__(self, config_gat):
        super(GAT, self).__init__()
        layer = GATLayer(config_gat)
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config_gat.num_layers)])

    def forward(self, querys, keys, values, attention_mask=None, position_weight = None):
        hidden_states = querys
        for layer_module in self.encoder:
            hidden_states = layer_module(querys, keys, values, attention_mask, position_weight)
        return hidden_states  # B, seq_len, D


def cosine_sim(im, s):
    return im.mm(s.t())

def pdist_cos(x1, x2):
    """
        compute cosine similarity between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise cosine distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_norm = x1 / x1.norm(dim=1)[:, None]
    x2_norm = x2 / x2.norm(dim=1)[:, None]
    res = torch.mm(x1_norm, x2_norm.transpose(0, 1))
    mask = torch.isnan(res)
    res[mask] = 0
    return res

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim

    def forward(self, im, s):
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_s = (self.margin + scores - d1).clamp(min=0)

        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


def get_optimizer(params, opt, t_total=-1):
    bertadam = BertAdam(params, lr=opt.learning_rate, warmup=opt.warmup, t_total=t_total)
    return bertadam


class Fusion(nn.Module):
    def __init__(self, opt):
        super(Fusion, self).__init__()
        self.f_size = opt.embed_size
        # self.gate0 = nn.Linear(self.f_size, self.f_size)
        # self.gate1 = nn.Linear(self.f_size, self.f_size)

        self.fusion1 = nn.Linear(self.f_size * 2, self.f_size)
        self.relu = nn.ReLU()
        self.fusion2 = nn.Linear(self.f_size, self.f_size)
        self.bn = nn.BatchNorm1d(self.f_size)
        # self.fusion3 = nn.Linear(self.f_size * 2, self.f_size)
        # self.sigmoid = torch.sigmoid()

    def forward(self, vec1, vec2):
        vec_cat = torch.cat([vec1, vec2], dim=-1)
        map1_out = self.fusion1(vec_cat)
        map1_out = self.relu(map1_out)
        map2_out = self.fusion2(map1_out)
        gate = torch.sigmoid(torch.mul(map2_out, vec2))
        gate_result = gate * vec2
        # concat_result = torch.cat([vec1, gate_result], dim=-1)
        # concat_result = self.relu(self.fusion3(concat_result))
        gate_result = self.bn(gate_result.permute(0, 2, 1)).permute(0, 2, 1)
        f = gate_result + vec1
        return f

class weightpool(nn.Module):
    def __init__(self, opt):
        super(weightpool, self).__init__()
        self.fc1 = nn.Linear(opt.embed_size, opt.embed_size)
        self.fc2 = nn.Linear(opt.embed_size, opt.embed_size)
        self.act = nn.ReLU()
    
    def forward(self, vec):
        out_features = self.fc1(vec)
        out_features = self.act(out_features)
        out_features = self.fc2(out_features)
        out_weights = nn.Softmax(dim=1)(out_features)
        out_emb = torch.mul(vec, out_weights)
        out_emb = out_emb.permute(0, 2, 1)
        pool_emb = torch.sum(out_emb.view(out_emb.size(0), out_emb.size(1), -1), dim=2)
        return pool_emb




class DSRAN(nn.Module):
    def __init__(self, opt):
        super(DSRAN, self).__init__()
        self.K = opt.K
        # self.img_enc = EncoderImageFull(opt)
        self.rcnn_enc = RcnnEncoder(opt)
        self.grid_enc = GridEncoder(opt)
        self.txt_enc = TextEncoder(opt)
        # config_rcnn = GATopt(opt.embed_size, 1)
        # config_img= GATopt(opt.embed_size, 1)
        # region_grid = GATopt(opt.embed_size, 1)
        # grid_region = GATopt(opt.embed_size, 1)
        config_cap= GATopt(opt.embed_size, 1)
        config_joint= GATopt(opt.embed_size, 1)
        # SSR
        # self.gat_1 = GAT(config_rcnn)
        # self.gat_2 = GAT(config_img)
        self.gat_cat = GAT(config_joint)
        # JSR
        self.gat_cat_1 = IGSAN(1, opt.embed_size, 8, is_share = False, drop = 0.2)
        self.gat_cat_2 = IGSAN(1, opt.embed_size, 8, is_share = False, drop = 0.2)
        self.fusion1 = Fusion(opt)
        self.fusion2 = Fusion(opt)
        self.gat_cap = GAT(config_cap)
        self.region_pool = weightpool(opt)
        self.grid_pool = weightpool(opt)
        self.cap_pool = weightpool(opt)
        # self.fusion3= nn.Linear(opt.embed_size, opt.embed_size)
        # self.fusion4 = nn.Linear(opt.embed_size, opt.embed_size)
        # self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(8)])
        
    def forward(self, img_rcnn, img_grid, img_mask, img_box, captions):

        n_regions = img_rcnn.shape[1]
        n_grids = img_grid.shape[1]
        bs = img_rcnn.shape[0]

        rcnn_emb = self.rcnn_enc(img_rcnn)
        rcnn_grid = self.grid_enc(img_grid)
        out_region = rcnn_emb
        out_grid = rcnn_grid
        attention_mask = img_mask.unsqueeze(1)
        # print(attention_mask.shape)

        tmp_mask = torch.ones(n_regions, n_regions, device=out_region.device).unsqueeze(0).unsqueeze(0)
        # print(tmp_mask.shape)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_regions * n_regions
        # print(tmp_mask.shape)
        region_aligns = (torch.cat([tmp_mask, attention_mask], dim=-1) == 0) # bs * 1 * n_regions *(n_regions+n_grids)
        # print(region_aligns[0][0][0])

        tmp_mask = torch.ones(n_grids, n_grids, device=out_grid.device).unsqueeze(0).unsqueeze(0)
        tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_grids * n_grids
        grid_aligns = (torch.cat([attention_mask.permute(0, 1, 3, 2), tmp_mask], dim=-1)==0) # bs * 1 * n_grids *(n_grids+n_regions)   
        # print(grid_aligns[0][0][0])

        region_alls = torch.cat([region_aligns, grid_aligns], dim=-2)


        out_all = torch.cat([out_region, out_grid], 1)
        out1 = self.gat_cat(out_all, out_all, out_all, region_alls)

        region_self = out1[:,:36,:]
        grid_self = out1[:,36:85,:]

        region_other = self.gat_cat_1(region_self, grid_self, grid_self)
        grid_other = self.gat_cat_2(grid_self, region_self, region_self)

        region_grid = self.fusion1(region_self, region_other)
        grid_region = self.fusion2(grid_self, grid_other)
        
        out_region_emb = self.region_pool(region_grid)
        out_grid_emb = self.grid_pool(grid_region)

        # img_gate = torch.sigmoid(self.fusion3(out_region_emb) + self.fusion4(out_grid_emb))
        img_cat = out_region_emb + out_grid_emb
        img_embs = l2norm(img_cat)

        cap_emb = self.txt_enc(captions)
        cap_gat = self.gat_cap(cap_emb, cap_emb, cap_emb)
        out_cap_emb = self.cap_pool(cap_gat)
        # out_cap_emb = torch.mean(out_cap_emb, 1)
        cap_embs = l2norm(out_cap_emb)

        return img_embs, cap_embs


class VSE(object):

    def __init__(self, opt):
        self.DSRAN = DSRAN(opt)
        self.DSRAN = nn.DataParallel(self.DSRAN)
        if torch.cuda.is_available():
            self.DSRAN.cuda()
            cudnn.benchmark = True
        self.criterion = ContrastiveLoss(margin=opt.margin)
        params = list(self.DSRAN.named_parameters())
        param_optimizer = params
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = opt.l_train * opt.num_epochs
        if opt.warmup == -1:
            t_total = -1
        self.optimizer = get_optimizer(params=optimizer_grouped_parameters, opt=opt, t_total=t_total)
        self.Eiters = 0

    def state_dict(self):
        state_dict = self.DSRAN.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.DSRAN.load_state_dict(state_dict)

    def train_start(self):
        self.DSRAN.train()

    def val_start(self):
        self.DSRAN.eval()

    def forward_emb(self, img_rcnn, img_grid, img_mask,  img_box, captions):
        if torch.cuda.is_available():
            img_rcnn = img_rcnn.cuda()
            img_grid = img_grid.cuda()
            img_mask = img_mask.cuda()
            # img_pos = img_pos.cuda()
            img_box = img_box.cuda()
            captions = captions.cuda()

        img_emb, cap_emb = self.DSRAN(img_rcnn, img_grid, img_mask, img_box, captions)

        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.data, img_emb.size(0))
        return loss

    def train_emb(self, img_rcnn, img_grid, img_mask,  img_box, captions, ids=None, *args):
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        img_emb, cap_emb = self.forward_emb(img_rcnn, img_grid, img_mask, img_box, captions)

        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)

        loss.backward()
        self.optimizer.step()
