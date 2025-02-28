#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/4/28 16:06
# @Author: ZhaoKe
# @File : deelenc.py
# @Software: PyCharm
import torch
import torchaudio


def simulated_AME(bs):
    return torch.rand(size=(bs, 8))


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_transform(x))


def SCDE2E_simulate_data(second=3, ):
    # second=3, seg_num=5, mel length=26, 128 130
    # second=5, seg_num=9, mel length=24, 128 216
    batch_size = 16
    sr = 22050
    seg_length = sr
    overlap = seg_length // 2
    seg_num = 1+(second-1)*2
    series_length = seg_length + overlap * (seg_num - 1)
    wav_sample_length = seg_length * seg_num
    print("second:{}, sr=seg_lenth:{}, seg_num:{}, \noverlap:{}, series_length:{}".format(second, seg_length, seg_num, overlap, series_length), wav_sample_length)

    batch_wav = torch.rand(size=(batch_size, wav_sample_length)).unsqueeze(1)
    wav2mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, hop_length=512, win_length=1024,
                                                   n_mels=128)
    # wav2mel = Wave2Mel(sr=22050)
    batch_mel = wav2mel(batch_wav)
    print("batch wav:{}, mel:{}.".format(batch_wav.shape, batch_mel.shape))
    print("mel length:{}.\n".format(batch_mel.shape[-1]/seg_num))

    y_lab = torch.randint(0, 5, size=(batch_size,))  #
    attri_1 = torch.randint(0, 2, size=(batch_size,))  # wet, dry
    attri_2 = torch.randint(0, 2, size=(batch_size,))  # wheeze true, false
    attri_3 = torch.randint(0, 2, size=(batch_size,))  # nose congest true, false
    attri_4 = torch.randint(0, 2, size=(batch_size,))  # Retch true, false
    prope_1 = torch.randint(0, 2, size=(batch_size,))  # male, female
    prope_2 = torch.randint(0, 80, size=(batch_size,))  # Age
    prope_3 = torch.randint(0, 2, size=(batch_size,))  # fever: True, False
    print("origin:", batch_wav.shape, batch_mel.shape, y_lab.shape, attri_1.shape, prope_1.shape)

    batch_aug_wav = batch_wav.view(batch_size * seg_num, seg_length)
    print("Reshape:", batch_aug_wav.shape)
    y_lab = y_lab.repeat_interleave(seg_num, dim=0)  #
    attri_1 = attri_1.repeat_interleave(seg_num, dim=0)  # wet, dry
    attri_2 = attri_2.repeat_interleave(seg_num, dim=0)  # wheeze true, false
    attri_3 = attri_3.repeat_interleave(seg_num, dim=0)  # nose congest true, false
    attri_4 = attri_4.repeat_interleave(seg_num, dim=0)  # Retch true, false
    prope_1 = prope_1.repeat_interleave(seg_num, dim=0)  # male, female
    prope_2 = prope_2.repeat_interleave(seg_num, dim=0)  # Age
    prope_3 = prope_3.repeat_interleave(seg_num, dim=0)  # fever: True, False
    print("aug:", batch_aug_wav.shape, batch_mel.shape, y_lab.shape, attri_1.shape, prope_1.shape)

    vae_latent1 = torch.rand(size=(batch_size * seg_num, 128))
    vae_latent2 = torch.rand(size=(batch_size * seg_num, 128))
    noise_latent = torch.rand(size=(batch_size * seg_num, 64))
    prope_latent = torch.rand(size=(batch_size * seg_num, 64))
    latent = torch.concat((vae_latent1, vae_latent2, noise_latent), dim=-1)
    print(latent.shape)
    latent_dim = (latent.shape[1]//seg_num)*(seg_num+1)
    pad_dim = latent_dim - latent.shape[1]
    left_pad, right_pad = pad_dim//2, pad_dim-pad_dim//2
    print(latent_dim, pad_dim, left_pad, right_pad)
    latent = torch.concat((torch.zeros(size=(batch_size*seg_num, left_pad)), latent, torch.zeros(size=(batch_size*seg_num, right_pad))), dim=-1)
    latent = latent.view(batch_size, seg_num, -1)
    print(latent.shape)
    print(latent.shape[1]/seg_num)

    from modules.attentions import MultiHeadAttention
    multihead_attn = MultiHeadAttention(embed_dim=latent.shape[-1], n_heads=2)
    # 执行多头注意力机制
    attn_mask = torch.zeros(batch_size, seg_num, seg_num, dtype=torch.float32)
    attn_output, attn_output_weights = multihead_attn(latent, latent, latent, attn_mask=attn_mask)  # 自注意力机制，Q, K, V 相同
    # 打印输出张量的形状
    print("输出张量 attn_output 的形状:", attn_output.shape)  # [2, 5, 8]
    print("注意力权重 attn_output_weights 的形状:", attn_output_weights.shape)  # [5, 2, 2]
    print("\n\n")
    # x_list = [torch.rand(sig_length) for _ in range(length)]
    # wav_inp = None
    # for item in x_list:
    #     if wav_inp is None:
    #         wav_inp = item
    #     else:
    #         wav_inp = torch.concat((wav_inp[:len(wav_inp) - overlap], item), dim=-1)


if __name__ == '__main__':
    SCDE2E_simulate_data(second=3)  # 26.0
    SCDE2E_simulate_data(second=4)  # 23.71428
    SCDE2E_simulate_data(second=5)  # 24.0
    SCDE2E_simulate_data(second=6)  # 23.5454
    SCDE2E_simulate_data(second=7)  # 23.2307

    # encoder = Wav2Vec(pretrained=True)
    # print(encoder)
    # # x = torch.randn(16, 16000)  # [1, 16000]
    # # z = encoder(x)  # [1, 512, 98]
    # # print(z.shape)
    # # x = torch.randn(1, 32000)  # [1, 32000]
    # # z = encoder(x)  # [1, 512, 198]
    # # print(z.shape)
    # x = torch.randn(16, 48000)  # [16, 48000]
    # z = encoder(x)  # [16, 512, 298]
    # print(z.shape)
