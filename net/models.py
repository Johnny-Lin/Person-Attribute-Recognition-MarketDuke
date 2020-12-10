import torch
from torch import nn
from torch.nn import init
from torchvision import models
from net.utils import ClassBlock
from torch.nn import functional as F


class Backbone_nFC(nn.Module):
    def __init__(self, class_num, model_name='resnet50_nfc'):
        super(Backbone_nFC, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num

        model_ft = getattr(models, self.backbone_name)(pretrained=True)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError
        
        self.class_filer = ClassBlock(input_dim=self.num_ftrs, class_num=self.class_num, activ='sigmoid')

        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.num_ftrs, class_num=1, activ='sigmoid') )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        #for i,pre in enumerate(pred_label):
        #    print(f'--->pred_label {i} size:',pre.size(),pre)
        #--->pred_label 0 size: torch.Size([1, 1]) tensor([[0.0152]], grad_fn=<SigmoidBackward>)
        #--->pred_label 1 size: torch.Size([1, 1]) tensor([[0.9873]], grad_fn=<SigmoidBackward>)
        #pred_label = torch.cat(pred_label, dim=1)
        #print(f'--->pred_label  size:',pred_label.size(),pred_label)
        #--->pred_label  size: torch.Size([1, 30]) 
        #tensor([[0.0152, 0.9873, 0.0118, 0.0156, 0.0244, 0.0528, 0.0190, 0.0149, 0.9737,
        # 0.9395, 0.9920, 0.0126, 0.9904, 0.0127, 0.5614, 0.0368, 0.0102, 0.0771,
        # 0.0130, 0.0149, 0.0146, 0.0219, 0.9387, 0.0116, 0.0141, 0.0948, 0.0120,
        # 0.0140, 0.0155, 0.0124]], grad_fn=<CatBackward>)
        pred_label = self.class_filer(x)
        return pred_label


class Backbone_nFC_Id(nn.Module):
    def __init__(self, class_num, id_num, model_name='resnet50_nfc_id'):
        super(Backbone_nFC_Id, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num
        self.id_num = id_num
        
        model_ft = getattr(models, self.backbone_name)(pretrained=True)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError
        
        for c in range(self.class_num+1):
            if c == self.class_num:
                self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, class_num=self.id_num, activ='none'))
            else:
                self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, class_num=1, activ='sigmoid'))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(self.class_num)]
        pred_label = torch.cat(pred_label, dim=1)
        pred_id = self.__getattr__('class_%d' % self.class_num)(x)
        return pred_label, pred_id

