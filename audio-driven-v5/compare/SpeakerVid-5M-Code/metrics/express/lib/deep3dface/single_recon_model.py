import torch
import torch.nn as nn
from lib.deep3dface.resnet import conv1x1, resnet50

class FaceReconModel(nn.Module):
    fc_dim=257
    def __init__(self, net_recon='resnet50'):
        super(FaceReconModel, self).__init__()
        last_dim = 2048
        backbone = resnet50(use_last_fc=False, num_classes=self.fc_dim)
        self.backbone = backbone
        
        self.final_layers = nn.ModuleList([
            conv1x1(last_dim, 80, bias=True), # id layer
            conv1x1(last_dim, 64, bias=True), # exp layer
            conv1x1(last_dim, 80, bias=True), # tex layer
            conv1x1(last_dim, 3, bias=True),  # angle layer
            conv1x1(last_dim, 27, bias=True), # gamma layer
            conv1x1(last_dim, 2, bias=True),  # tx, ty
            conv1x1(last_dim, 1, bias=True)   # tz
        ])
        for m in self.final_layers:
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.backbone(x)
        output = []
        for layer in self.final_layers:
            output.append(layer(x))
        x = torch.flatten(torch.cat(output, dim=1), 1)
        return x
