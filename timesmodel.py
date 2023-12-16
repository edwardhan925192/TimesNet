import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math

#Positional embedding used in transformer
class PositionalEmbedding(nn.Module):
    '''
    1. Positional information is added on top of embedded data
    '''
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()

#gradient descent set to False
        pe.require_grad = False

#Unsqueezing(1) = Adding dimension
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

#Not updated but go together
        self.register_buffer('pe', pe)

#Every torch class need forward its where the actual process occurs
    def forward(self, x):
        return self.pe[:, :x.size(1)]

# ================= Token embedding ================= #
class TokenEmbedding(nn.Module):

    '''
    each are single tokens
    [t11, t12, t13]
    [t21, t22, t23]

    1. Takes (batchsize, sequence_length, channels)
    2. permute for 1d conv (batchsize, channels, sequence_length)
    3. embed whole features into d_model dimension
    4. return (batchsize, sequence_length, d_channels)
    '''

    def __init__(self, c_in, d_model):
      super(TokenEmbedding, self).__init__()
      padding = 1 if torch.__version__ >= '1.5.0' else 2
      self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                  kernel_size=3, padding=padding, padding_mode='circular', bias=False)

  #initializing
      for m in self.modules():
        if isinstance(m, nn.Conv1d):
          nn.init.kaiming_normal_(
              m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
      x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
      return x

# Everything is added
class DataEmbedding(nn.Module):
    '''
    1. Wrapper for Positional, 1d token embedding.
    '''
    def __init__(self, c_in, d_model, dropout=0.1): #embed_type='fixed', freq='h',
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x): #x_mark
      # if x_mark is None:
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

#Positional embedding is broad casted when used in torch
class Inception_Block_V1(nn.Module):
    '''
    1. After reshaping 1D data to 2D with FFT it goes over convolution layers
    2. It goes through different kernel_size appended in the module lists
    3. Averaged and returned
    '''
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []

        # =========== Kernel size increases ========= #
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))

        # ============ Average them ============ #
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []

        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

def FFT_for_Period(x, k=2):
    '''
    There are index amount of fft periods for the data
    '''
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)

    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k                                

        # =============== ANALYZATION BACKBONE ================ #

        # parameter-efficient design
        if configs.cnn_type == 'inception':
          self.conv = nn.Sequential(
              Inception_Block_V1(configs.d_model, configs.d_ff,
                                num_kernels=configs.num_kernels),
              nn.GELU(),
              Inception_Block_V1(configs.d_ff, configs.d_model,
                                num_kernels=configs.num_kernels)
          )
        # ================ DECONFORMABLE BLOCK ================ # 
        if configs.cnn_type == 'dcvn':
          self.conv = nn.Sequential(
              DeconformableBlock(configs.d_model, configs.d_ff,
                                num_kernels=configs.num_kernels),
              nn.GELU(),
              DeconformableBlock(configs.d_ff, configs.d_model,
                                num_kernels=configs.num_kernels)
          )

    def forward(self, x):
        B, T, N = x.size()

        # ============= 2. Finding Amplitudes and periods ============= #
        period_list, period_weight = FFT_for_Period(x, self.k)

        #Res list
        res = []
        for i in range(self.k):
            period = period_list[i]

            # =========== 3. Reshaping (padding) seq_len // period =========== #
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # =========== 3. Reshaping for 2D (periods * frequency) =========== #
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()

            # =========== 4. 2D conv =========== #
            out = self.conv(out)

            # =========== 4. reshape back =========== #
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)

            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)

        # =========== 5. adaptive aggregation =========== #
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
        return res

# ===================== MODEL MAIN ========================== #

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.target_col = configs.target_col
        #e_layers number of timeblock module saved in model "list"
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
				#embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, #configs.embed, configs.freq,
                                           configs.dropout)
				#e_layers
        self.layer = configs.e_layers

        #within each batch normalizing each features
        self.layer_norm = nn.LayerNorm(configs.d_model)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

    def forecast(self, x_enc):
        # Calculate means and stdev
        # ============== Non stationary normalization ================== #
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc_normalized = x_enc

        # embedding
        enc_out = self.enc_embedding(x_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension


        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                  dec_out = self.forecast(x_enc)
                  return dec_out[:, -self.pred_len:, self.target_col]  # [B, L, D]
