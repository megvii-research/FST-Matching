#!/usr/bin/env python3
import megengine as mge
import megengine.functional as F
import megengine.hub as hub
import megengine.module as M

import math
from .resnet import resnet18
from .se_module import SELayer

class FSTNet(M.Module):

    def __init__(self, num_classes=2):

        super(FSTNet, self).__init__()

        self.source_encoder = resnet18(pretrained=True)
        self.target_encoder = resnet18(pretrained=True)
        self.source_se = SELayer(self.source_encoder.out_num_features)
        self.target_se = SELayer(self.target_encoder.out_num_features)
        self.feature_dim = self.source_encoder.out_num_features + self.target_encoder.out_num_features
        self.fc = M.Linear(self.feature_dim, num_classes)

    def se(self, x, se_type):
        if se_type == 'source':
            x, x_unrelated = self.source_se(x)
            x_unrelated = F.flatten(x_unrelated, 1)
            x = F.flatten(x, 1)
            x  = self.source_encoder.fc(x)
        elif se_type == 'target':
            x, x_unrelated = self.target_se(x)
            x_unrelated = F.flatten(x_unrelated, 1)
            x = F.flatten(x, 1)
            x = self.target_encoder.fc(x)
        else:
            raise NotImplementedError
        return x_unrelated, x

    def forward(self, x):
        source_feat = self.source_encoder(x, extract_feat=True)
        souce_unrelated, source_result = self.se(source_feat, se_type="source")
        target_feat = self.target_encoder(x, extract_feat=True)
        target_unrelated, target_result = self.se(target_feat, se_type="target")

        feat_mask_target = F.concat((target_unrelated * 0, souce_unrelated), axis=1)
        feat_mask_source = F.concat((target_unrelated, souce_unrelated * 0), axis=1)
        feat_mask_all = F.concat((target_unrelated * 0, souce_unrelated * 0), axis=1)
        feat = F.concat((target_unrelated, souce_unrelated), axis=1)

        det_result = self.fc(feat)
        interaction_result = self.fc(feat) - self.fc(feat_mask_target) - self.fc(feat_mask_source) + self.fc(feat_mask_all)

        if self.training:
            return det_result, source_result, target_result, interaction_result
        return det_result


# vim: ts=4 sw=4 sts=4 expandtab
