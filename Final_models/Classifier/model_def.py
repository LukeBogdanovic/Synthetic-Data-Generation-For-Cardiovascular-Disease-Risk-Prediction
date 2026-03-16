import torch
import torch.nn as nn

class ECG_Conv_Block(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad):
        super(ECG_Conv_Block, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, padding=pad)
        self.batch_norm = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout1d(0.2)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.max_pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, ecg):
        x = self.conv(ecg)
        x = self.batch_norm(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.max_pool(x)
        return x


class Classifier(nn.Module):

    def __init__(self, num_leads, num_risk_factors, num_classes):
        super(Classifier, self).__init__()
        chs = [num_leads, 32, 64, 128, 256]
        k_sizes = [5, 7, 9, 11]
        blocks = []
        for cin, cout, k_size in zip(chs[:-1], chs[1:], k_sizes):
            pad = (k_size-1) // 2
            blocks.append(ECG_Conv_Block(cin, cout, k_size, pad))
        self.conv_blocks = nn.Sequential(*blocks)
        ecg_feat_dim = 256
        risk_emb_dim = 64
        if num_risk_factors > 0:
            self.risk_mlp = nn.Sequential(
                nn.Linear(num_risk_factors, 128),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, risk_emb_dim),
                nn.LeakyReLU(0.3, inplace=True)
            )
        else:
            self.risk_mlp = None
            risk_emb_dim = 0

        self.head = nn.Sequential(
            nn.Linear(ecg_feat_dim + risk_emb_dim, 128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, ecg, risk_factors=None):
        x = self.conv_blocks(ecg)
        x = x.mean(dim=-1)
        if self.risk_mlp is not None:
            if risk_factors is None:
                raise ValueError(
                    "Risk factors must be provided when num_risk_factors > 0")
            r = self.risk_mlp(risk_factors)
            x = torch.cat([x, r], dim=1)
        logits = self.head(x)
        return logits

def build_classifier(num_leads: int = 3, num_risk_factors: int = 7, num_classes: int = 4) -> nn.Module:
    return Classifier(num_leads=num_leads, num_risk_factors=num_risk_factors, num_classes=num_classes)