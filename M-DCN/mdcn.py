import torch
import torch.nn
from torch.nn import functional as F


class MDCN(torch.nn.Module):
    def __init__(self, data, ent_dim, rel_dim, **kwargs):
        super(MDCN, self).__init__()
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.reshape_H = 20
        self.reshape_W = 20
        self.out_1 = 8
        self.out_2 = 20
        self.out_3 = 8

        self.in_channels = kwargs["in_channels"]
        self.out_channels = kwargs["out_channels"]
        self.filt_height = kwargs["filt_height"]
        self.filt_width = kwargs["filt_width"]

        self.entity_embedding = torch.nn.Embedding(data.entities_num, ent_dim, padding_idx=0)
        self.relation_embedding = torch.nn.Embedding(data.relations_num, rel_dim, padding_idx=0)
        filter1_dim = self.in_channels * self.out_1 * 1 * 5
        self.filter1 = torch.nn.Embedding(data.relations_num, filter1_dim, padding_idx=0)
        filter2_dim = self.in_channels * self.out_2 * 3 * 3
        self.filter3 = torch.nn.Embedding(data.relations_num, filter2_dim, padding_idx=0)
        filter3_dim = self.in_channels * self.out_3 * 1 * 9
        self.filter5 = torch.nn.Embedding(data.relations_num, filter3_dim, padding_idx=0)

        self.input_drop = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_drop = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.feature_map_drop = torch.nn.Dropout2d(kwargs["feature_map_dropout"])

        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.bn1 = torch.nn.BatchNorm2d(self.out_1 + self.out_2 + self.out_3)

        self.bn1_1 = torch.nn.BatchNorm2d(self.out_1)
        self.bn1_2 = torch.nn.BatchNorm2d(self.out_2)
        self.bn1_3 = torch.nn.BatchNorm2d(self.out_3)
        self.bn2 = torch.nn.BatchNorm1d(ent_dim)

        self.loss = torch.nn.BCELoss()
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(data.entities_num)))

        fc_length = self.reshape_H * self.reshape_W * (self.out_1 + self.out_2 + self.out_3)
        self.fc = torch.nn.Linear(fc_length, ent_dim)

    def init(self):
        torch.nn.init.xavier_normal_(self.entity_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.relation_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.filter1.weight.data)
        torch.nn.init.xavier_normal_(self.filter3.weight.data)
        torch.nn.init.xavier_normal_(self.filter5.weight.data)

    def forward(self, entity_id, relation_id):
        # (b, 1, 200)
        entity = self.entity_embedding(entity_id).reshape(-1, 1, self.ent_dim)
        relation = self.relation_embedding(relation_id).reshape(-1, 1, self.rel_dim)
        f1 = self.filter1(relation_id)
        f1 = f1.reshape(entity.size(0) * self.in_channels * self.out_1, 1, 1, 5)
        f3 = self.filter3(relation_id)
        f3 = f3.reshape(entity.size(0) * self.in_channels * self.out_2, 1, 3, 3)
        f5 = self.filter5(relation_id)
        f5 = f5.reshape(entity.size(0) * self.in_channels * self.out_3, 1, 1, 9)

        # (b, 2, 200)→ (b, 200, 2)→ (b, 1, 20, 20)
        x = torch.cat([entity, relation], 1).transpose(1, 2).reshape(-1, 1, self.reshape_H, self.reshape_W)
        x = self.bn0(x)
        x = self.input_drop(x)

        # (1 ,b, 20, 20)
        x = x.permute(1, 0, 2, 3)

        # (1, b*in*out, H-kH+1, W-kW+1)
        x1 = F.conv2d(x, f1, groups=entity.size(0), padding=(0, 2))
        x1 = x1.reshape(entity.size(0), self.out_1, self.reshape_H, self.reshape_W)
        x1 = self.bn1_1(x1)

        x3 = F.conv2d(x, f3, groups=entity.size(0), padding=(1, 1))
        x3 = x3.reshape(entity.size(0), self.out_2, self.reshape_H, self.reshape_W)
        x3 = self.bn1_2(x3)

        x5 = F.conv2d(x, f5, groups=entity.size(0), padding=(0, 4))
        x5 = x5.reshape(entity.size(0), self.out_3, self.reshape_H, self.reshape_W)
        x5 = self.bn1_3(x5)

        x = torch.cat([x1, x3, x5], dim=1)
        x = torch.relu(x)
        x = self.feature_map_drop(x)

        # (b, fc_length)
        x = x.view(entity.size(0), -1)

        # (b, ent_dim)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.relu(x)

        # (batch, ent_dim)*(ent_dim, ent_num)=(batch, ent_num)
        x = torch.mm(x, self.entity_embedding.weight.transpose(1, 0))
        x += self.bias.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
