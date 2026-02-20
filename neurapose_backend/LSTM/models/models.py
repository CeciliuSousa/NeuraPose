# ================================================================
# neurapose_backend/LSTM/models/models.py
# ================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

########################################

def ensure_BTF(x):
    # Aceita entradas em (B, C, T, J) e converte para (B, T, F),
    # onde F = C * J, para padronizar o formato (batch, time, features)
    # usado pelos modelos sequenciais abaixo.
    # Útil quando os dados vêm como tensors 4D (ex.: skeleton data, sensores por junta, etc.).
    if x.ndim == 4:
        # Entrada: (B, C, T, V)
        # Permuta para (B, T, C, V) para alinhar com o treino
        x = x.permute(0, 2, 1, 3) 
        # Achata para (B, T, C*V)
        x = x.reshape(x.size(0), x.size(1), -1)
    assert x.dim() == 3, f"Esperado (B, T, F), mas recebi {x.shape}"
    return x

# =============================================================================
# SimpleLSTM
# -----------------------------------------------------------------------------
# O que é:
#   - Um LSTM unidirecional padrão (PyTorch) seguido de uma camada linear.
# Como funciona:
#   - Processa a sequência temporal (B, T, F) e usa o último hidden state
#     (h_n[-1]) como sumarização da sequência para classificação.
# Quando usar:
#   - Dados sequenciais onde dependências de longo prazo importam.
# Referências:
#   - LSTM original: Hochreiter & Schmidhuber (1997) https://www.bioinf.jku.at/publications/older/2604.pdf
#   - PyTorch nn.LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
#   - Implementações de referência (GitHub; uso aplicado com LSTM/BiLSTM):
#       Flair (NLP aplicado): https://github.com/flairNLP/flair
#       AllenNLP (framework de pesquisa aplicada): https://github.com/allenai/allennlp
#       TorchText examples (benchmarks): https://github.com/pytorch/text/tree/main/examples
#       SpeechBrain (ASR e afins): https://github.com/speechbrain/speechbrain
# =============================================================================
class LSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = ensure_BTF(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


# =============================================================================
# RobustLSTM (BiLSTM com dropout)
# -----------------------------------------------------------------------------
# O que é:
#   - Um LSTM bidirecional com múltiplas camadas e dropout, seguido por MLP.
# Como funciona:
#   - A concatenação dos últimos hidden states de forward e backward (h_n[-2], h_n[-1])
#     dá uma representação que captura contexto passado e futuro.
# Quando usar:
#   - Dados ruidosos e/ou com dependências bidirecionais (contexto de ambos os lados).
# Referências:
#   - PyTorch nn.LSTM bidirecional: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
#   - Implementações de referência (GitHub; uso aplicado com BiLSTMs):
#       Flair: https://github.com/flairNLP/flair
#       AllenNLP: https://github.com/allenai/allennlp
#       SpeechBrain: https://github.com/speechbrain/speechbrain
# =============================================================================
class RobustLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = ensure_BTF(x)
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(h_n)


# =============================================================================
# PooledLSTM (BiLSTM + pooling temporal)
# -----------------------------------------------------------------------------
# O que é:
#   - BiLSTM seguido de um pooling temporal (AdaptiveMaxPool1d) para comprimir
#     a informação ao longo de T, e um MLP para classificação.
# Como funciona:
#   - Em vez de usar apenas o último estado, agrega todos os passos com max-pooling,
#     capturando picos/atividades salientes.
# Quando usar:
#   - Quando eventos podem ocorrer em qualquer posição temporal e max-pooling
#     é uma boa sumarização.
# Referências:
#   - Pooling adaptativo no PyTorch: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html
#   - Implementações de referência (GitHub; uso aplicado de poolings sobre sequências):
#       Sentence-Transformers: https://github.com/UKPLab/sentence-transformers
#       OpenNMT-py (seq2seq real): https://github.com/OpenNMT/OpenNMT-py
# =============================================================================
class PooledLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = ensure_BTF(x)
        out, _ = self.lstm(x)        # (B, T, 2H)
        out = out.permute(0, 2, 1)   # (B, 2H, T)
        pooled = self.pool(out).squeeze(-1)
        return self.fc(pooled)


# =============================================================================
# BILSTM (BiLSTM + MLP)
# -----------------------------------------------------------------------------
# O que é:
#   - BiLSTM padrão com cabeça MLP, similar ao RobustLSTM porém sem dropout final.
# Como funciona:
#   - Concatena os últimos estados de ambas as direções e passa ao MLP.
# Quando usar:
#   - Baseline forte para tarefas sequenciais com dependências bidirecionais.
# Referências:
#   - PyTorch nn.LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
#   - Implementações de referência (GitHub; uso aplicado com BiLSTMs):
#       Sentence-Transformers: https://github.com/UKPLab/sentence-transformers
#       Flair: https://github.com/flairNLP/flair
# =============================================================================
class BILSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=128, num_layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = ensure_BTF(x)
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.fc(h_n)


########################################
# Attention LSTM "mais original" (Bahdanau-like)
########################################

# =============================================================================
# BahdanauAttention (atenção aditiva)
# -----------------------------------------------------------------------------
# O que é:
#   - Mecanismo de atenção aditiva de Bahdanau et al. ("Neural Machine Translation
#     by Jointly Learning to Align and Translate").
# Como funciona:
#   - Calcula scores com v^T tanh(W_h * h_t + W_s * s) para cada passo t,
#     normaliza com softmax e faz soma ponderada.
# Quando usar:
#   - Quando é útil destacar partes específicas da sequência (alinhamento).
# Referências:
#   - Paper: https://arxiv.org/abs/1409.0473
#   - Implementações de referência (GitHub; atenção aditiva aplicada):
#       OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
#       JoeyNMT: https://github.com/joeynmt/joeynmt
# =============================================================================
class BahdanauAttention(nn.Module):
    # Atenção aditiva: score = v^T tanh(W_h*h_t + W_s*s)
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H, s):
        # H: (B, T, H), s: (B, H) (contexto/estado de consulta)
        # Expand s sobre T
        s_exp = self.W_s(s).unsqueeze(1)          # (B,1,H)
        scores = self.v(torch.tanh(self.W_h(H) + s_exp))  # (B,T,1)
        attn = torch.softmax(scores, dim=1)       # (B,T,1)
        context = torch.sum(attn * H, dim=1)      # (B,H)
        return context, attn


# =============================================================================
# AttentionLSTM (BiLSTM + Bahdanau)
# -----------------------------------------------------------------------------
# O que é:
#   - Uma BiLSTM que produz estados por passo (H) e usa BahdanauAttention
#     com a "consulta" sendo a concatenação dos últimos estados (fwd/bwd).
# Como funciona:
#   - A atenção foca timesteps importantes e gera um "context vector"
#     passado a um MLP para classificação.
# Quando usar:
#   - Eventos localizados ou segmentos críticos na sequência.
# Referências:
#   - Atenção Bahdanau: https://arxiv.org/abs/1409.0473
#   - LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
#   - Implementações de referência (GitHub; Bahdanau attention aplicada):
#       OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
#       JoeyNMT: https://github.com/joeynmt/joeynmt
# =============================================================================
class AttentionLSTM(nn.Module):
    # BiLSTM + Bahdanau attention sobre H com consulta s = concat(h_fwd_last, h_bwd_last)
    def __init__(self, input_size=34, hidden_size=128, num_layers=1, num_classes=2, dropout=0.0):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = BahdanauAttention(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = ensure_BTF(x)
        H, (h_n, _) = self.bilstm(x)       # H: (B,T,2H), h_n: (2L,B,H)
        s = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B,2H)
        context, _ = self.attn(H, s)       # (B,2H)
        return self.fc(context)


########################################
# TCN (Bai et al. 2018) com blocos residuais causais dilatados
########################################

# =============================================================================
# Chomp1d
# -----------------------------------------------------------------------------
# O que é:
#   - Camada utilitária que remove o excesso de padding à direita para manter
#     a causalidade após convoluções com padding.
# Referência:
#   - TCN (repo oficial) e uso aplicado:
#       LocusLab TCN: https://github.com/locuslab/TCN
#       tsai (biblioteca de séries): https://github.com/timeseriesAI/tsai
# =============================================================================
class Chomp1d(nn.Module):
    # Remove o padding à direita para manter causalidade
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


# =============================================================================
# TemporalBlock
# -----------------------------------------------------------------------------
# O que é:
#   - Bloco TCN com duas convs 1D causais dilatadas, ReLU, dropout e conexão residual.
# Como funciona:
#   - Dilatações crescentes expandem o campo receptivo temporal sem aumentar muito
#     a profundidade. A conexão residual estabiliza o treinamento.
# Referências:
#   - Paper TCN: Bai, Kolter, Koltun (2018) https://arxiv.org/abs/1803.01271
#   - Implementações de referência (GitHub; TCN aplicado):
#       LocusLab TCN: https://github.com/locuslab/TCN
#       tsai: https://github.com/timeseriesAI/tsai
# =============================================================================
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                            padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                            padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        the_relu2 = nn.ReLU()
        self.relu2 = the_relu2
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# =============================================================================
# TCN (pilha de TemporalBlocks)
# -----------------------------------------------------------------------------
# O que é:
#   - Implementação de TCN com dilatações exponenciais [1,2,4,8,...],
#     pooling global e cabeça MLP para classificação.
# Quando usar:
#   - Padrões temporais locais/estacionários, séries com larga escala temporal.
# Referências:
#   - Paper: https://arxiv.org/abs/1803.01271
#   - Implementações de referência (GitHub; TCN aplicado):
#       LocusLab TCN: https://github.com/locuslab/TCN
#       tsai: https://github.com/timeseriesAI/tsai
#       PyTorch Forecasting: https://github.com/zalandoresearch/pytorch-forecasting
# =============================================================================
class TCN(nn.Module):
    # Pilha de TemporalBlocks com dilatações [1, 2, 4, 8, ...]
    def __init__(self, input_size=34, channels=(64, 64, 64, 64), kernel_size=3, dropout=0.2, num_classes=2, **kwargs):
        super().__init__()
        # Compatibilidade: se num_channels for passado (estilo antigo),
        # converta para tuple repetida e respeite kernel_size do kwargs.
        num_channels = kwargs.pop("num_channels", None)
        ks_compat = kwargs.pop("kernel_size", None)
        if num_channels is not None:
            channels = tuple([num_channels] * 4)  # 4 níveis padrão
        if ks_compat is not None:
            kernel_size = ks_compat

        layers = []
        num_levels = len(channels)
        in_ch = input_size
        for i in range(num_levels):
            out_ch = channels[i]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = ensure_BTF(x)
        x = x.permute(0, 2, 1)  # (B, F, T)
        y = self.tcn(x)         # (B, C, T)
        y = self.pool(y).squeeze(-1)
        return self.head(y)


########################################
# Transformer "mais original" (Vaswani et al.)
# - Embeddings posicionais
# - Token [CLS] no início
########################################

# =============================================================================
# PositionalEncoding (seno/cosseno fixo)
# -----------------------------------------------------------------------------
# O que é:
#   - Codificação posicional determinística com seno/cosseno do "Attention is All You Need".
# Por que:
#   - O Transformer é permutacionalmente invariante; posições são necessárias
#     para modelar ordem.
# Referência:
#   - Paper: Vaswani et al. (2017) https://arxiv.org/abs/1706.03762
#   - Implementações de referência (GitHub; uso aplicado do positional encoding):
#       PyTorch examples: https://github.com/pytorch/examples/tree/main/word_language_model
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (L, D)
        position = torch.arange(0, max_len).unsqueeze(1)  # (L,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: (d_model // 2)])
        self.register_buffer('pe', pe)  # não treina

    def forward(self, x):
        # x: (T, B, D)
        T = x.size(0)
        return x + self.pe[:T].unsqueeze(1)  # (T,B,D)


# =============================================================================
# TransformerModel (Encoder com token [CLS])
# -----------------------------------------------------------------------------
# O que é:
#   - Transformer Encoder padrão (PyTorch) com posição senoidal e token [CLS]
#     para classificação.
# Como funciona:
#   - Projeta features para d_model, adiciona [CLS] no início, aplica encoder,
#     usa o vetor do [CLS] na cabeça MLP.
# Quando usar:
#   - Dependências de longo alcance e padrões complexos não locais.
# Referências:
#   - Paper: https://arxiv.org/abs/1706.03762
#   - PyTorch Transformer: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
#   - Implementações de referência (GitHub; classificação com [CLS] em escala):
#       HuggingFace Transformers: https://github.com/huggingface/transformers
#       TimesFM (Transformers para séries): https://github.com/google-research/timesfm
#       PyTorch examples: https://github.com/pytorch/examples/tree/main/word_language_model
# =============================================================================
class TransformerModel(nn.Module):
    def __init__(self, input_size=34, hidden_size=128, num_layers=2, num_classes=2, num_heads=8, dropout=0.3):
        super().__init__()
        self.d_model = hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = ensure_BTF(x)               # (B,T,F)
        x = self.input_proj(x)          # (B,T,D)
        B, T, D = x.shape

        # prepend CLS
        cls = self.cls_token.expand(-1, B, -1)    # (1,B,D)
        x = x.permute(1, 0, 2)                    # (T,B,D)
        x = torch.cat([cls, x], dim=0)            # (T+1,B,D)

        x = self.pos_encoder(x)
        z = self.encoder(x)                       # (T+1,B,D)
        cls_out = z[0]                            # (B,D)
        return self.head(cls_out)


########################################
# Temporal Fusion Transformer (TFT) - versão compacta e mais fiel
########################################

# =============================================================================
# GLU (Gated Linear Unit)
# -----------------------------------------------------------------------------
# O que é:
#   - Unidade com porta (gate) que controla o fluxo de informação
#     via a * sigmoid(b).
# Referência:
#   - GLU: Dauphin et al. (2017) https://arxiv.org/abs/1612.08083
#   - Implementações de referência (GitHub; GLU em aplicações seq2seq/conv):
#       fairseq: https://github.com/facebookresearch/fairseq
# =============================================================================
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(d, 2 * d)

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


# =============================================================================
# GatedResidualNetwork (GRN)
# -----------------------------------------------------------------------------
# O que é:
#   - Bloco com projeções, ELU, GLU, residual e layer norm. Componente-chave no TFT.
# Referência:
#   - TFT: Lim et al. (2020) https://arxiv.org/abs/1912.09363
#   - Implementação referência (Google Research): https://github.com/google-research/google-research/tree/master/tft
#   - Implementações de referência (GitHub; TFT aplicado):
#       PyTorch Forecasting: https://github.com/zalandoresearch/pytorch-forecasting
# =============================================================================
class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(d_in, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_out)
        self.glu = GLU(d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x):
        y = F.elu(self.lin1(x))
        y = self.dropout(self.lin2(y))
        y = self.glu(y)
        y = self.norm(self.skip(x) + y)
        return y


# =============================================================================
# VariableSelectionNetwork (VSN)
# -----------------------------------------------------------------------------
# O que é:
#   - Módulo que projeta cada variável e aprende pesos de importância
#     para seleção ponderada de variáveis ao longo do tempo.
# Por que:
#   - Interpretabilidade e foco em variáveis relevantes em cada timestep.
# Referências:
#   - TFT: https://arxiv.org/abs/1912.09363
#   - Implementações de referência (GitHub; TFT aplicado):
#       PyTorch Forecasting: https://github.com/zalandoresearch/pytorch-forecasting
#       Darts: https://github.com/unit8co/darts
# =============================================================================
class VariableSelectionNetwork(nn.Module):
    # Seleção ponderada de variáveis (concatena variáveis já embutidas)
    def __init__(self, d_in, d_model, n_vars):
        super().__init__()
        self.n_vars = n_vars
        self.var_projs = nn.ModuleList([nn.Linear(d_in, d_model) for _ in range(n_vars)])
        self.softmax = nn.Softmax(dim=-1)
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x_list):
        # x_list: lista de tensores [B, T, d_in] ou [B, d_in] (broadcast T se necessário)
        proj = []
        attn_scores = []
        T = None
        for xi, proj_i in zip(x_list, self.var_projs):
            if xi.dim() == 2:
                # estático: (B, d_in) -> (B, 1, d_in) para broadcast
                xi = xi.unsqueeze(1)
            if T is None:
                T = xi.size(1)
            yi = proj_i(xi)                  # (B, T, d_model)
            proj.append(yi)
            attn_scores.append(self.attn(yi))  # (B, T, 1)

        Y = torch.stack(proj, dim=-2)          # (B, T, n_vars, d_model)
        A = torch.stack(attn_scores, dim=-2)   # (B, T, n_vars, 1)
        w = self.softmax(A)                    # (B, T, n_vars, 1)
        out = (w * Y).sum(dim=-2)              # (B, T, d_model)
        return out, w.squeeze(-1)              # (B, T, d_model), (B, T, n_vars)


# =============================================================================
# TemporalFusionTransformer (compacto e didático)
# -----------------------------------------------------------------------------
# O que é:
#   - Versão compacta inspirada no TFT com VSN temporal, GRNs e encoder-decoder
#     com atenção multi-head. Fornece essência do TFT para classificação.
# Limitações:
#   - Não é 1:1 com o paper (faltam diversas partes, como seleção de variáveis
#     estáticas/futuras separadas, quantile forecasts, etc.).
# Quando usar:
#   - Quando deseja incorporar seleção de variáveis e atenção temporal de forma
#     mais interpretável que um Transformer puro.
# Referências:
#   - Paper TFT: https://arxiv.org/abs/1912.09363
#   - Implementações de referência (GitHub; TFT aplicado):
#       PyTorch Forecasting: https://github.com/zalandoresearch/pytorch-forecasting
#       Darts: https://github.com/unit8co/darts
#       Google Research TFT: https://github.com/google-research/google-research/tree/master/tft

# https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py
# =============================================================================
class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                input_size=34,
                d_model=128,
                n_heads=8,
                num_encoder_layers=2,
                num_decoder_layers=1,
                dropout=0.1,
                num_classes=2):
        super().__init__()
        # Supõe todas as F variáveis como "observadas" temporais. Para demo,
        # não diferenciamos estáticas/observadas/covariadas futuramente, mas
        # mantemos VSN e GRNs essenciais.

        self.input_size = input_size
        self.d_model = d_model
        self.norm_in = nn.LayerNorm(d_model)
        self.pos_scale = 0.1

        # Embedding por variável (linear)
        self.var_emb = nn.Linear(1, d_model)

        # Variable Selection Network (temporal) — assume todas as F variáveis
        self.vsn_temporal = VariableSelectionNetwork(d_in=d_model, d_model=d_model, n_vars=input_size)

        # Encoder-Decoder com atenção
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Gated skip connections (GRNs)
        self.grn_enc = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.grn_dec = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # Cabeça de classificação (pooling + MLP)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # Positional encoding (aprendida simples)
        self.pos = nn.Parameter(torch.zeros(1, 4096, d_model)) # máx T=1000 (ajuste conforme)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        # x: (B,T,F)
        x = ensure_BTF(x)
        B, T, F = x.shape
        if T > self.pos.size(1):
            raise RuntimeError(f"T={T} > max_pos_length={self.pos.size(1)}; aumente a matriz posicional.")

        # Embedding por variável: processa cada feature separadamente como (B,T,1)->(B,T,D)
        vars_emb = []
        for fi in range(F):
            vi = x[..., fi:fi+1]            # (B,T,1)
            ei = self.var_emb(vi)           # (B,T,D)
            vars_emb.append(ei)

        # Variable selection temporal
        z_t, _ = self.vsn_temporal(vars_emb)  # (B,T,D)

        # Adiciona posição
        z_t = self.norm_in(z_t + self.pos_scale * self.pos[:, :T, :])        # (B,T,D)

        # Encoder
        enc_out = self.encoder(z_t)           # (B,T,D)
        enc_out = self.grn_enc(enc_out)       # gated residual

        # Decoder com consulta do último passo (one-step "query").
        # Para classificação de sequência, podemos usar um token de consulta (B,1,D)
        query = enc_out[:, -1:, :]            # usa última posição como query
        dec_out = self.decoder(query, enc_out)  # (B,1,D)
        dec_out = self.grn_dec(dec_out)

        # Pooling global do encoder + decoder e soma (estratégia simples)
        enc_pooled = self.pool(enc_out.transpose(1, 2)).squeeze(-1)  # (B,D)
        dec_vec = dec_out.squeeze(1)                                  # (B,D)
        fused = 0.5 * (enc_pooled + dec_vec)

        return self.head(fused)


########################################
# WaveNet original-like: causal dilated conv + gated residual blocks
########################################

# =============================================================================
# GatedResidualBlock (núcleo do WaveNet)
# -----------------------------------------------------------------------------
# O que é:
#   - Bloco com duas convoluções paralelas (filter/gate) com função de ativação
#     tanh e sigmoid, combinadas multiplicativamente (gated), além de conexões
#     residual e skip.
# Como funciona:
#   - Convoluções causais dilatadas aumentam o campo receptivo; as conexões
#     residual/skip ajudam no fluxo de gradiente.
# Referências:
#   - WaveNet paper: van den Oord et al. (2016) https://arxiv.org/abs/1609.03499
#   - Implementações de referência (GitHub; WaveNet aplicado):
#       r9y9/wavenet_vocoder: https://github.com/r9y9/wavenet_vocoder
#       NVIDIA NeMo (vocoder WaveNet-like): https://github.com/NVIDIA/NeMo
# =============================================================================
class GatedResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.filter = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp = Chomp1d(padding)
        self.res = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        f = torch.tanh(self.chomp(self.filter(x)))
        g = torch.sigmoid(self.chomp(self.gate(x)))
        z = f * g
        z = self.dropout(z)
        res = self.res(z) + x
        skip = self.skip(z)
        return res, skip


# =============================================================================
# WaveNet (para classificação)
# -----------------------------------------------------------------------------
# O que é:
#   - Pilha de GatedResidualBlocks com convoluções causais dilatadas (dilations),
#     projeções 1x1 e pooling global para produzir logits de classe.
# Quando usar:
#   - Sinais temporais com estruturas hierárquicas/longo alcance (áudio, sensores, etc.).
# Referências:
#   - Paper: https://arxiv.org/abs/1609.03499
#   - Implementações de referência (GitHub; WaveNet aplicado):
#       r9y9/wavenet_vocoder: https://github.com/r9y9/wavenet_vocoder
#       ESPnet (vocoders WaveNet-like): https://github.com/espnet/espnet
#       NVIDIA NeMo: https://github.com/NVIDIA/NeMo
# =============================================================================
class WaveNet(nn.Module):
    def __init__(self, input_size=34, channels=64, kernel_size=2, dilations=(1, 2, 4, 8, 16, 32), num_classes=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.input_proj = nn.Conv1d(input_size, channels, 1)

        self.blocks = nn.ModuleList([
            GatedResidualBlock(channels, kernel_size, d, dropout) for d in dilations
        ])
        self.relu = nn.ReLU()
        self.out1 = nn.Conv1d(channels, channels, 1)
        self.out2 = nn.Conv1d(channels, num_classes, 1)

        # Para classificação, faremos pooling global no tempo do mapa final
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = ensure_BTF(x)          # (B,T,F)
        B, T, F = x.shape
        if F != self.input_size:
            raise RuntimeError(f"WaveNet: input_size esperado = {self.input_size}, mas F={F}. Ajuste input_size.")
        x = x.permute(0, 2, 1)     # (B,F,T)
        x = self.input_proj(x)     # (B,C,T)

        skip_sum = None
        for blk in self.blocks:
            x, skip = blk(x)
            skip_sum = skip if skip_sum is None else (skip_sum + skip)

        y = self.relu(skip_sum)
        y = self.relu(self.out1(y))
        y = self.out2(y)           # (B,num_classes,T)
        y = self.pool(y).squeeze(-1)  # (B,num_classes)
        return y