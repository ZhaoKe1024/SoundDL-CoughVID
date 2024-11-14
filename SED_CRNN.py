#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/11/13 20:57
# @Author: ZhaoKe
# @File : SED_CRNN.py
# @Software: PyCharm
import torch
from torch import nn

"""
def get_model(data_in, data_out, dropout_rate, nb_cnn2d_filt, pool_size,
                                rnn_size, fnn_size, classification_mode, weights):
    # model definition
    spec_start = Input(shape=(data_in[-3], data_in[-2], data_in[-1]))
    spec_cnn = spec_start
    for i, convCnt in enumerate(pool_size):
        spec_cnn = Conv2D(filters=nb_cnn2d_filt, kernel_size=(3, 3), padding='same')(spec_cnn)
        spec_cnn = BatchNormalization()(spec_cnn)
        spec_cnn = Activation('relu')(spec_cnn)
        spec_cnn = MaxPooling2D(pool_size=(1, pool_size[i]))(spec_cnn)
        spec_cnn = Dropout(dropout_rate)(spec_cnn)
    spec_cnn = Permute((2, 1, 3))(spec_cnn)

    spec_rnn = Reshape((data_in[-2], -1))(spec_cnn)
    for nb_rnn_filt in rnn_size:
        spec_rnn = Bidirectional(
            GRU(nb_rnn_filt, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate,
                return_sequences=True),
            merge_mode='mul'
        )(spec_rnn)

    doa = spec_rnn
    for nb_fnn_filt in fnn_size:
        doa = TimeDistributed(Dense(nb_fnn_filt))(doa)
        doa = Dropout(dropout_rate)(doa)

    doa = TimeDistributed(Dense(data_out[1][-1]))(doa)
    doa = Activation('tanh', name='doa_out')(doa)

    # SED
    sed = spec_rnn
    for nb_fnn_filt in fnn_size:
        sed = TimeDistributed(Dense(nb_fnn_filt))(sed)
        sed = Dropout(dropout_rate)(sed)
    sed = TimeDistributed(Dense(data_out[0][-1]))(sed)
    sed = Activation('sigmoid', name='sed_out')(sed)

    model = Model(inputs=spec_start, outputs=[sed, doa])
    model.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mse'], loss_weights=weights)

    model.summary()
    return model
"""


class CRNN(nn.Module):
    def __init__(self, params=None):
        super().__init__()
        pool_size = params["pool_size"]
        dropout_rate = params["dropout_rate"]
        nn_cnn2d_filt = params["nb_cnn2d_filt"]
        rnn_size = params["rnn_size"]
        fnn_size = params["fnn_size"]
        inp, oup = 1, nn_cnn2d_filt
        self.feature_extractor = nn.Sequential()
        print("Build CRNN Model:")
        pool_cnt = 2
        for idx, convCnt in enumerate(pool_size):
            print("Layer:", idx)
            self.feature_extractor.append(nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=(3, 3), padding=1))
            self.feature_extractor.append(nn.BatchNorm2d(num_features=oup))
            self.feature_extractor.append(nn.ReLU())
            if pool_cnt > 0:
                self.feature_extractor.append(nn.MaxPool2d((1, convCnt)))
                self.feature_extractor.append(nn.Dropout())
                pool_cnt -= 1
            inp, oup = oup, 64
        # rnn_module = nn.Sequential()
        # for nb in rnn_size:
        #     rnn_module.append(nn.GRU())
        # fnn_module = nn.Sequential()
        # for nb in fnn_size:
        #     fnn_module.append()

    def forward(self, x_mel):
        # out = self.feature_extractor(x_mel)
        for layer in self.feature_extractor:
            out = layer(x_mel)
            # print(out.shape)
            x_mel = out
        return x_mel


def main():
    # pool_size = [8, 8, 2]
    params = {"pool_size": [8, 8, 2], "dropout_rate": 0.0, "batch_size": 32, "nb_cnn2d_filt": 64,
              "rnn_size": [128, 128], "fnn_size": [128]}
    crnn_model = CRNN(params=params)
    print(crnn_model)
    x = torch.rand(size=(32, 1, 128, 64))
    crnn_model(x)
    # loss_f1 = nn.CrossEntropyLoss()
    # loss_f2 = nn.MSELoss()
    # optim = torch.optim.Adam()
    print(crnn_model)


if __name__ == '__main__':
    main()
