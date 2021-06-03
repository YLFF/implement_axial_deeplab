import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torchsummary import summary
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY,Backbone,ShapeSpec
from detectron2.modeling.backbone.resnet import (
    BasicStem,
    BottleneckBlock,
    DeformBottleneckBlock,
    ResNet,
)
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
from torchsummary import summary
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone.resnet import (
    BasicStem,
    BottleneckBlock,
    DeformBottleneckBlock,
    ResNet,
)
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

class DeepLabStem(CNNBlockBase):    # classic 7*7 implement as 3 3*3
    """
    The DeepLab ResNet stem (layers before the first residual block).
    """

    def __init__(self, in_channels=3, out_channels=128, norm="BN"):
        """
        Args:
            norm (str or callable): norm after the first conv layer.
                See :func:`layers.get_norm` for supported format.
        """
        super().__init__(in_channels, out_channels, 4)
        self.in_channels = in_channels
        self.conv1 = Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2),
        )
        self.conv2 = Conv2d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels // 2),
        )
        self.conv3 = Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )
        weight_init.c2_msra_fill(self.conv1)
        weight_init.c2_msra_fill(self.conv2)
        weight_init.c2_msra_fill(self.conv3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu_(x)
        x = self.conv2(x)
        x = F.relu_(x)
        x = self.conv3(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


@BACKBONE_REGISTRY.register()
def build_axial_deeplab_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """
    # need registration of new blocks/stems?
    norm = cfg.MODEL.RESNETS.NORM
    if cfg.MODEL.RESNETS.STEM_TYPE == "basic":
        stem = BasicStem(
            in_channels=input_shape.channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )
    elif cfg.MODEL.RESNETS.STEM_TYPE == "deeplab":
        stem = DeepLabStem(
            in_channels=input_shape.channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )
    elif cfg.MODEL.RESNETS.SETM_TYPE == "AxialBlock":
        stem = DeepLabStem(
            in_channels=input_shape.channels,
            out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
            norm=norm,
        )
        print('not implemented!')
    else:
        raise ValueError("Unknown stem type: {}".format(
            cfg.MODEL.RESNETS.STEM_TYPE))

    # fmt: off
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res4_dilation = cfg.MODEL.RESNETS.RES4_DILATION
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    res5_multi_grid = cfg.MODEL.RESNETS.RES5_MULTI_GRID
    # fmt: on
    # cut ASPP part
    '''
    assert res4_dilation in {1, 2}, "res4_dilation cannot be {}.".format(res4_dilation)
    assert res5_dilation in {1, 2, 4}, "res5_dilation cannot be {}.".format(res5_dilation)
    if res4_dilation == 2:
        # Always dilate res5 if res4 is dilated.
        assert res5_dilation == 4
    '''
    num_blocks_per_stage = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }[depth]

    stages = []

    # Avoid creating variables without gradients
    # It consumes extra memory and may cause allreduce to fail
    out_stage_idx = [{
        "res2": 2,
        "res3": 3,
        "res4": 4,
        "res5": 5
    }[f] for f in out_features]  #2,3,5
    max_stage_idx = max(out_stage_idx)  #5
    for idx, stage_idx in enumerate(range(2, max_stage_idx + 1)):
        if stage_idx == 4:
            dilation = res4_dilation
        elif stage_idx == 5:
            dilation = res5_dilation
        else:
            dilation = 1
        first_stride = 1 if idx == 0 or dilation > 1 else 2  #dilation: [1,1,dilation4,dilation5] first_stride=[1,2,x,1]
        stage_kargs = {
            "num_blocks":
            num_blocks_per_stage[idx],
            "stride_per_block":
            [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            "in_channels":
            in_channels,
            "out_channels":
            out_channels,
            "norm":
            norm,
        }
        stage_kargs["bottleneck_channels"] = bottleneck_channels
        stage_kargs["stride_in_1x1"] = stride_in_1x1
        stage_kargs["dilation"] = dilation
        stage_kargs["num_groups"] = num_groups
        if deform_on_per_stage[idx]:
            stage_kargs["block_class"] = AxialBlock
            stage_kargs["deform_modulated"] = deform_modulated
            stage_kargs["deform_num_groups"] = deform_num_groups
        else:
            stage_kargs["block_class"] = BottleneckBlock
        if stage_idx == 5:
            stage_kargs['block_class']=AxialBlock
            '''stage_kargs.pop("dilation")
            stage_kargs["dilation_per_block"] = [
                dilation * mg for mg in res5_multi_grid
            ]'''  #mg=[1,2,4]   eg dilation5=4, dilationperblock=[4,8,16]
            blocks=ResNet.make_stage(AxialBlock,num_blocks=num_blocks_per_stage[idx],in_channels=in_channels,out_channels=out_channels)
        else:
            blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features).freeze(freeze_at)


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""
    def __init__(self,in_planes,out_planes):
        self.in_planes=in_planes
        self.out_planes=out_planes
        super(qkv_transform,self).__init__(in_channels=self.in_planes,out_channels=self.out_planes,kernel_size=1)
        self.qkv_transform=nn.Conv1d(self.in_planes,self.out_planes,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=False,padding_mode='zeros')    
    def forward(self,x):
        out=self.qkv_transform(x)
        return out




def downsampling(inplanes,outplanes,stride):
    return nn.Sequential(conv1x1(inplanes, outplanes, stride=stride))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def pooling(size):
    return nn.Upsample((size,size))
def up_sample(size):
    return nn.Upsample((size,size*2))
class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=64,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups #8
        self.group_planes = out_planes // groups #16
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.upsample=pooling(kernel_size)
        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, )
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        #self.bn_qk = nn.BatchNorm2d(groups)
        #self.bn_qr = nn.BatchNorm2d(groups)
        #self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0) 
        key_index = torch.arange(kernel_size).unsqueeze(1) 
        relative_index = key_index - query_index + kernel_size - 1 #(56,56)
        self.register_buffer('flatten_index', relative_index.view(-1)) #(ks*ks)
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        #didnt implement local constraint
        N, W, C, H = x.shape
        if W>C:
            x=self.upsample(x)
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape

        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x)) #(n*w,2c,h)
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)
        
        # q,k,v is transform of x, with shape(n*w,g,g_plane,h),(n*w,g,h),(n*w,g,g_plane*2,h),
        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        #q,k,v_embedding.shape=[g_channel,ks,ks],[g_channel,ks,ks],[g_channel*2,ks,ks],
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding) #shape(b*w,g,ks,ks)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        #stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        #nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(CNNBlockBase):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=8,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=64,norm=None):
        super().__init__(in_channels,out_channels,1)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.))
        self.pool=pooling(kernel_size)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, out_channels * self.expansion,stride=(2,1))
        self.bn2 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.up_sample=up_sample(kernel_size)
        self.stride = stride
        self.downsample = downsampling(self.in_channels,self.out_channels,self.stride)
    def forward(self, x):
        identity = x # (32,64)
        
        #out = self.pool(x) #64*64
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out) #(64,64)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        if x.shape[2]==64:
            out=self.up_sample(out)
        out += identity
        out = self.relu(out)

        return out
