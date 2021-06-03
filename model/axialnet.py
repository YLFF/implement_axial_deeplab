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


__all__ = ['axial26s', 'axial50s', 'axial50m', 'axial50l']
'''utils:qkvtransform'''
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








def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=64,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

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
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
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


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=64):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride, width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
 
        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AxialAttentionNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=0.5):
        super(AxialAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res2 = self._make_layer(block, int(128 * s), layers[0], kernel_size=56)
        self.res3 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=56,
                                       dilate=replace_stride_with_dilation[0])
        self.res4 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=28,
                                       dilate=replace_stride_with_dilation[1])
        self.res5 = self._make_layer(block, int(1024 * s), layers[3], stride=2, kernel_size=14,
                                       dilate=replace_stride_with_dilation[2])
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2=nn.Conv2d(int(1024*s*2),2048,1,1,0,1,1,False)
        self.bn3=norm_layer(2048)

        for m in self.modules():
            #print(m)
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if isinstance(m, qkv_transform):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AxialBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, 
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x=self.conv2(x)
        x=self.bn3(x)
        #x = self.avgpool(x)



        return x


    def forward(self, x):
        return self._forward_impl(x)


def axial26s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [1, 2, 4, 1], s=0.5, **kwargs)
    return model


def axial50s(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.5, **kwargs)
    return model


def axial50m(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.75, **kwargs)
    return model


def axial50l(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=1, **kwargs)
    return model


#@BACKBONE_REGISTRY.register()
def build_axial_deeplab_backbone(cfg,input_shape):
    norm = cfg.MODEL.RESNETS.NORM
    if cfg.MODEL.RESNETS.STEM_TYPE=='deeplab':
        stem=DeepLabStem(input_shape.channels,cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,norm)
    elif cfg.MODEL.RESNETS.STEM_TYPE=='axial':
        stem=DeepLabStem(input_shape.channels,cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,norm)
        print('not implemented yet')
    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    #res4_dilation       = cfg.MODEL.RESNETS.RES4_DILATION
    #res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION
    deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    res5_multi_grid     = cfg.MODEL.RESNETS.RES5_MULTI_GRID
    num_block_per_stage={50:[3,4,6,3],101:[3,4,23,3]}[depth]

    stages=[]
    out_stage_idx = [{"res2": 2, "res3": 3, "res4": 4, "res5": 5}[f] for f in out_features]  #2,3,5
    max_stage_idx = max(out_stage_idx) #5
    for idx,stage_idx in enumerate(range(2,max_stage_idx+1)):
        #idx,stage_idx=[(0,2),(1,3),(2,4),(3,5)]
        dilation=1
        first_stride=1

class AxialBackbone(Backbone):
    def __init__(self,cfg,input_shape):
        super().__init__()
    def forward(self,x):
        return axial50s(Ture)
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
            
        }


if __name__=='__main__':
    model=axial50l(True)
    #print(axial50s(True))
    #print(model)
    model=model.cuda()
    summary(model,(3,224,224),8)

