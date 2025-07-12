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

class ConditionalTCNAutoencoder(nn.Module):
    def __init__(self, m_dim=11, fault_dim=13, channels=[32,64,64], kernel_size=3):
        super().__init__()
        self.fault_dim = fault_dim
        
        # 라벨을 임베딩으로 변환하는 레이어
        self.fault_embedding = nn.Sequential(
            nn.Linear(1, 32),  # 단일 라벨값을 32차원으로
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64)
        )
        
        # Encoder
        enc_layers = []
        in_ch = m_dim
        for i,ch in enumerate(channels):
            enc_layers.append(TCNBlock(in_ch, ch, kernel_size, dilation=2**i))
            in_ch = ch
        self.encoder = nn.Sequential(*enc_layers)
        
        # 조건부 정보 결합을 위한 MLP
        self.condition_mlp = nn.Sequential(
            nn.Linear(channels[-1] + 64, channels[-1]),  # fault embedding(64) + 마지막 채널
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
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

    def forward(self, m_seq, fault_labels, fault_time=None):
        """
        Args:
            m_seq: (B, 50, 11) 조작변수 시퀀스
            fault_labels: (B,) fault 라벨 (0-12 범위의 정수)
            fault_time: (B,) fault 발생 시점 (옵션)
        """
        # m_seq: (B, 50, 11) → (B, 11, 50)
        x = m_seq.transpose(1, 2)
        
        # 라벨을 시퀀스 길이로 확장 (train_model.py와 동일한 방식)
        fault_labels_seq = fault_labels.unsqueeze(1).expand(-1, m_seq.size(1))  # shape: (B, seq_len)
        
        # 각 시점의 라벨을 float로 변환하고 차원 추가
        fault_labels_float = fault_labels_seq.float().unsqueeze(-1)  # (B, seq_len, 1)
        
        # Fault 조건 임베딩 (시퀀스의 각 시점별로)
        if fault_time is not None:
            # fault_time을 0-1로 정규화하고 fault_labels와 결합
            norm_time = fault_time.float() / m_seq.size(1)
            norm_time_seq = norm_time.unsqueeze(1).expand(-1, m_seq.size(1))  # (B, seq_len)
            fault_cond = torch.cat([fault_labels_float, norm_time_seq.unsqueeze(-1)], dim=-1)  # (B, seq_len, 2)
        else:
            fault_cond = fault_labels_float  # (B, seq_len, 1)
            
        # 각 시점별로 임베딩
        B, seq_len, cond_dim = fault_cond.size()
        fault_cond_flat = fault_cond.reshape(-1, cond_dim)  # (B*seq_len, cond_dim)
        fault_emb_flat = self.fault_embedding(fault_cond_flat)  # (B*seq_len, 64)
        fault_emb = fault_emb_flat.reshape(B, seq_len, 64)  # (B, seq_len, 64)
        
        # 인코더 통과
        z = self.encoder(x)  # (B, 64, T)
        
        # 시퀀스의 각 시점에 조건부 정보 결합
        B, C, T = z.size()
        z_flat = z.transpose(1, 2).reshape(-1, C)  # (B*T, C)
        fault_emb_flat = fault_emb.reshape(-1, 64)  # (B*seq_len, 64)
        
        # 조건부 정보와 특징 결합
        z_cond = torch.cat([z_flat, fault_emb_flat], dim=-1)  # (B*T, C+64)
        z_cond = self.condition_mlp(z_cond)  # (B*T, C)
        z_cond = z_cond.reshape(B, T, -1).transpose(1, 2)  # (B, C, T)
        
        # 디코더 통과
        rec = self.decoder(z_cond)
        
        # 시퀀스 길이 맞추기
        if rec.size(2) != m_seq.size(1):
            rec = rec[:, :, :m_seq.size(1)]
        
        # (B, 11, 50) → (B, 50, 11)
        return rec.transpose(1, 2)