import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        # 패딩 크기 계산: (kernel_size-1) * dilation
        pad = (kernel_size-1) * dilation
        
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        
        # Chomp 함수 정의 (패딩된 부분 제거)
        self.chomp = lambda x: x[:, :, :-pad] if pad != 0 else x
        
    def forward(self, x):
        # 네트워크 통과
        out = self.net(x)
        
        # 잔차 연결 준비
        res = x if self.downsample is None else self.downsample(x)
        
        # out과 res의 길이를 강제로 맞춰줌 (중첩된 padding 효과 처리)
        min_len = min(out.size(2), res.size(2))
        out = out[:, :, :min_len]
        res = res[:, :, :min_len]
        
        return nn.functional.leaky_relu(out + res, 0.1)

class MTCNAutoencoder(nn.Module):
    def __init__(self, m_dim=11, channels=[32,64,64], kernel_size=3):
        super().__init__()
        # Encoder
        enc_layers = []
        in_ch = m_dim
        for i,ch in enumerate(channels):
            enc_layers.append(TCNBlock(in_ch, ch, kernel_size, dilation=2**i))
            in_ch = ch
        self.encoder = nn.Sequential(*enc_layers)
        
        # Decoder (ConvTranspose1d)
        dec_layers = []
        rev_channels = list(reversed(channels))
        in_ch = rev_channels[0]
        for out_ch in rev_channels[1:]:
            dec_layers += [
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size, padding=1),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2)
            ]
            in_ch = out_ch
        # 마지막 복원
        dec_layers += [
            nn.ConvTranspose1d(in_ch, m_dim, kernel_size, padding=1),
        ]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, m_seq):
        # m_seq: (B, 50, 11) → (B, 11, 50)
        x = m_seq.transpose(1, 2)
        
        # 인코더 통과
        z = self.encoder(x)
        
        # 디코더 통과
        rec = self.decoder(z)
        
        # 시퀀스 길이 맞추기
        if rec.size(2) != m_seq.size(1):
            rec = rec[:, :, :m_seq.size(1)]
        
        # (B, 11, 50) → (B, 50, 11)
        return rec.transpose(1, 2)