# -*- coding: utf-8 -*-
# @Author: wangdongdong
# @Time: 2024/1/15 10:10

import glob
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import tensorflow as tf
from tensorflow.keras.models import Model
import tensorflow.keras as keras
from tensorflow.keras.layers import Lambda, Input, BatchNormalization, Concatenate, LayerNormalization, PReLU, Layer, Conv2D, Conv2DTranspose
import soundfile as sf
import numpy as np
from tensorflow.keras import backend
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(4)
blockLen = 400
blockshift = 200
win = np.sin(np.arange(.5, blockLen - .5 + 1) / blockLen * np.pi)
win = tf.constant(win, dtype='float32')
input_norm = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DprnnBlock_stateful(Layer):

    def __init__(self, numUnits, batch_size, L, width, channel, causal=True, **kwargs):
        super(DprnnBlock_stateful, self).__init__(**kwargs)

        self.numUnits = numUnits
        self.batch_size = batch_size
        self.causal = causal
        self.intra_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.numUnits // 2,
                                                                      return_sequences=True,
                                                                      implementation=2,
                                                                      recurrent_activation='hard_sigmoid',
                                                                      unroll=True))

        self.intra_fc = keras.layers.Dense(units=self.numUnits, )

        if self.causal:
            self.intra_ln = keras.layers.LayerNormalization(center=True, scale=True, axis=[-1, -2])
        else:
            self.intra_ln = keras.layers.LayerNormalization(center=False, scale=False)

        self.inter_rnn = keras.layers.LSTM(units=self.numUnits,
                                           return_sequences=True,
                                           implementation=2,
                                           recurrent_activation='hard_sigmoid',
                                           unroll=True,
                                           stateful=False,
                                           return_state=True)

        self.inter_fc = keras.layers.Dense(units=self.numUnits, )

        if self.causal:
            self.inter_ln = keras.layers.LayerNormalization(center=True, scale=True, axis=[-1, -2])
        else:
            self.inter_ln = keras.layers.LayerNormalization(center=False, scale=False)

        self.L = L
        self.width = width
        self.channel = channel

    def call(self, x):
        # Intra-Chunk Processing
        batch_size = self.batch_size
        L = self.L
        width = self.width

        intra_rnn = self.intra_rnn
        intra_fc = self.intra_fc
        intra_ln = self.intra_ln
        inter_rnn = self.inter_rnn
        inter_fc = self.inter_fc
        inter_ln = self.inter_ln
        channel = self.channel
        causal = self.causal
        h_state = tf.Variable(tf.zeros([batch_size, width, channel]), trainable=False, )
        c_state = tf.Variable(tf.zeros([batch_size, width, channel]), trainable=False, )

        # input shape (bs,T,F,C) --> (bs*T,F,C)
        intra_LSTM_input = tf.reshape(x, [-1, width, channel])
        # (bs*T,F,C)
        intra_LSTM_out = intra_rnn(intra_LSTM_input)

        # (bs*T,F,C) channel axis dense
        intra_dense_out = intra_fc(intra_LSTM_out)

        if causal:
            # (bs*T,F,C) --> (bs,T,F,C) Freq and channel norm
            intra_ln_input = tf.reshape(intra_dense_out, [batch_size, -1, width, channel])
            intra_out = intra_ln(intra_ln_input)

        else:
            # (bs*T,F,C) --> (bs,T*F*C) global norm
            intra_ln_input = tf.reshape(intra_dense_out, [batch_size, -1])
            intra_ln_out = intra_ln(intra_ln_input)
            intra_out = tf.reshape(intra_ln_out, [batch_size, L, width, channel])

        # (bs,T,F,C)
        intra_out = keras.layers.Add()([x, intra_out])
        # %%
        # (bs,T,F,C) --> (bs,F,T,C)
        inter_LSTM_input = tf.transpose(intra_out, [0, 2, 1, 3])
        # (bs,F,T,C) --> (bs*F,T,C)
        inter_LSTM_input = tf.reshape(inter_LSTM_input, [batch_size * width, L, channel])

        inter_LSTM_out, inter_LSTM_h, inter_LSTM_c = inter_rnn(inter_LSTM_input, initial_state=[h_state[0], c_state[0]])

        # (bs,F,T,C) Channel axis dense
        inter_dense_out = inter_fc(inter_LSTM_out)

        inter_dense_out = tf.reshape(inter_dense_out, [batch_size, width, L, channel])

        if causal:
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_ln_input = tf.transpose(inter_dense_out, [0, 2, 1, 3])
            inter_out = inter_ln(inter_ln_input)

        else:
            # (bs,F,T,C) --> (bs,F*T*C)
            inter_ln_input = tf.reshape(inter_dense_out, [batch_size, -1])
            inter_ln_out = inter_ln(inter_ln_input)
            inter_out = tf.reshape(inter_ln_out, [batch_size, width, L, channel])
            # (bs,F,T,C) --> (bs,T,F,C)
            inter_out = tf.transpose(inter_out, [0, 2, 1, 3])
        # (bs,T,F,C)
        inter_out = keras.layers.Add()([intra_out, inter_out])

        with tf.control_dependencies([inter_LSTM_h, inter_LSTM_c]):
            self.update_h = h_state.assign(tf.expand_dims(inter_LSTM_h, axis=0), read_value=False)
            self.update_c = c_state.assign(tf.expand_dims(inter_LSTM_c, axis=0), read_value=False)

        self.reset_h = h_state.assign(tf.zeros([batch_size, width, channel]), read_value=False)
        self.reset_c = c_state.assign(tf.zeros([batch_size, width, channel]), read_value=False)

        tf.add_to_collection('upop', [self.update_h, self.update_c])
        tf.add_to_collection('rsop', [self.reset_h, self.reset_c])

        return inter_out

class Conv2D_stateful(Conv2D):
    def __init__(self, filters, keranel_size, strides, padding, state_shape, *args, **kwargs):
        self.state_shape = state_shape
        super().__init__(filters, keranel_size, strides, padding, *args, **kwargs)

    # Override call
    def call(self, x):
        padding_state = tf.Variable(tf.zeros(self.state_shape), trainable=False)
        padding_data = tf.concat([padding_state, x], axis=1)
        output = super().call(padding_data)

        with tf.control_dependencies([padding_data, output]):
            self.update_op = padding_state.assign(x, read_value=False)

        self.reset = padding_state.assign(tf.zeros(self.state_shape), read_value=False)
        tf.add_to_collection('upop', self.update_op)
        tf.add_to_collection('rsop', self.reset)
        return output

class DeConv2D_stateful(Conv2DTranspose):

    def __init__(self, filters, keranel_size, strides, padding, state_shape, *args, **kwargs):
        self.state_shape = state_shape
        self.padding = padding
        super().__init__(filters, keranel_size, strides, padding, *args, **kwargs)

    # Override call
    def call(self, x):
        padding_state = tf.Variable(tf.zeros(self.state_shape), trainable=False)
        padding_data = tf.concat([padding_state, x], axis=1)
        output = super().call(padding_data)

        if self.padding == 'same':
            output = output[:, -1:, :, :]

        with tf.control_dependencies([padding_data, output]):
            self.update_op = padding_state.assign(x, read_value=False)

        self.reset = padding_state.assign(tf.zeros(self.state_shape), read_value=False)

        tf.add_to_collection('upop', self.update_op)
        tf.add_to_collection('rsop', self.reset)
        return output


class Noise_Suppression_Model:
    def __init__(self):
        self.CNN_filter_list = [32, 32, 32, 64, 128]
        self.DeCNN_filter_list = [64, 32, 32, 32, 2]
        self.CNN_state_list = [[1, 1, 201, 2], [1, 1, 100, 32], [1, 1, 50, 32], [1, 1, 50, 32], [1, 1, 50, 64]]
        self.DeCNN_state_list = [[1, 1, 50, 256], [1, 1, 50, 128], [1, 1, 50, 64], [1, 1, 50, 64], [1, 1, 100, 64]]

    def overlapadd(self, frame, hop=200):
        n_frame, l_frame = frame.shape
        length = l_frame + (n_frame - 1) * hop
        output = np.zeros(length)
        for i in range(n_frame):
            output[hop * i: hop * i + l_frame] += frame[i]
        return output

    def stftLayer(self, x, mode='mag_pha', stateful=False):
        if not stateful:
            frames = tf.signal.frame(x, blockLen, blockshift)
        else:
            frames = tf.expand_dims(x, axis=1)

        frames = win * frames
        stft_dat = tf.signal.rfft(frames)
        output_list = []
        if mode == 'mag_pha':
            mag = tf.math.abs(stft_dat)
            phase = tf.math.angle(stft_dat)
            output_list = [mag, phase]
        elif mode == 'real_imag':
            real = tf.math.real(stft_dat)
            imag = tf.math.imag(stft_dat)
            output_list = [real, imag]

        return output_list

    def ifftLayer(self, x, mode='mag_pha'):
        if mode == 'mag_pha':
            s1_stft = (tf.cast(x[0], tf.complex64) * tf.exp((1j * tf.cast(x[1], tf.complex64))))
        elif mode == 'real_imag':
            s1_stft = tf.cast(x[0], tf.complex64) + 1j * tf.cast(x[1], tf.complex64)

        return tf.signal.irfft(s1_stft)

    def overlapAddLayer(self, x):
        x = x - tf.expand_dims(tf.reduce_mean(x, axis=-1), 2)

        return tf.signal.overlap_and_add(x, blockshift)

    def mk_mask(self, x):
        [noisy_real, noisy_imag, mask] = x
        mask_real = mask[:, :, :, 0]
        mask_imag = mask[:, :, :, 1]
        enh_real = noisy_real * mask_real - noisy_imag * mask_imag
        enh_imag = noisy_real * mask_imag + noisy_imag * mask_real

        return [enh_real, enh_imag]

    def rnn_encoder(self,):
        Encoder_input = Input(batch_shape=(1, 400))
        real, imag = Lambda(self.stftLayer, arguments={'mode': 'real_imag', 'stateful': True})(Encoder_input)
        input_complex_spec = tf.stack([real, imag], axis=-1)

        if input_norm:
            input_complex_spec = LayerNormalization(axis=[-1, -2])(input_complex_spec)

        name = 'encoder'
        conv_1 = Conv2D_stateful(filters=self.CNN_filter_list[0],
                                 keranel_size=(2, 5),
                                 strides=(1, 2),
                                 padding=[[0, 0], [0, 0], [0, 2], [0, 0]],
                                 state_shape=self.CNN_state_list[0],
                                 name=name + '_conv_1')(input_complex_spec)
        bn_1 = BatchNormalization(name=name + '_bn_1')(conv_1)
        out_1 = PReLU(shared_axes=[1, 2])(bn_1)

        conv_2 = Conv2D_stateful(filters=self.CNN_filter_list[1],
                                 keranel_size=(2, 3),
                                 strides=(1, 2),
                                 padding=[[0, 0], [0, 0], [0, 1], [0, 0]],
                                 state_shape=self.CNN_state_list[1],
                                 name=name + '_conv_2')(out_1)
        bn_2 = BatchNormalization(name=name + '_bn_2')(conv_2)
        out_2 = PReLU(shared_axes=[1, 2])(bn_2)

        conv_3 = Conv2D_stateful(filters=self.CNN_filter_list[2],
                                 keranel_size=(2, 3),
                                 strides=(1, 1),
                                 padding=[[0, 0], [0, 0], [1, 1], [0, 0]],
                                 state_shape=self.CNN_state_list[2],
                                 name=name + '_conv_3')(out_2)
        bn_3 = BatchNormalization(name=name + '_bn_3')(conv_3)
        out_3 = PReLU(shared_axes=[1, 2])(bn_3)

        conv_4 = Conv2D_stateful(filters=self.CNN_filter_list[3],
                                 keranel_size=(2, 3),
                                 strides=(1, 1),
                                 padding=[[0, 0], [0, 0], [1, 1], [0, 0]],
                                 state_shape=self.CNN_state_list[3],
                                 name=name + '_conv_4')(out_3)
        bn_4 = BatchNormalization(name=name + '_bn_4')(conv_4)
        out_4 = PReLU(shared_axes=[1, 2])(bn_4)

        conv_5 = Conv2D_stateful(filters=self.CNN_filter_list[4],
                                 keranel_size=(2, 3),
                                 strides=(1, 1),
                                 padding=[[0, 0], [0, 0], [1, 1], [0, 0]],
                                 state_shape=self.CNN_state_list[4],
                                 name=name + '_conv_5')(out_4)
        bn_5 = BatchNormalization(name=name + '_bn_5')(conv_5)
        out_5 = PReLU(shared_axes=[1, 2])(bn_5)

        dp_in = out_5
        for i in range(2):
            dp_in = DprnnBlock_stateful(numUnits=128, batch_size=1, L=-1, width=50, channel=128, causal=True)(dp_in)
        dp_out = dp_in

        name = 'decoder'
        skipcon_1 = Concatenate(axis=-1)([out_5, dp_out])

        deconv_1 = DeConv2D_stateful(filters=self.DeCNN_filter_list[0],
                                     keranel_size=(2, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     state_shape=self.DeCNN_state_list[0],
                                     name=name + '_dconv_1')(skipcon_1)
        dbn_1 = BatchNormalization(name=name + '_dbn_1')(deconv_1)
        dout_1 = PReLU(shared_axes=[1, 2])(dbn_1)

        skipcon_2 = Concatenate(axis=-1)([out_4, dout_1])

        deconv_2 = DeConv2D_stateful(filters=self.DeCNN_filter_list[1],
                                     keranel_size=(2, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     state_shape=self.DeCNN_state_list[1],
                                     name=name + '_dconv_2')(skipcon_2)
        dbn_2 = BatchNormalization(name=name + '_dbn_2')(deconv_2)
        dout_2 = PReLU(shared_axes=[1, 2])(dbn_2)

        skipcon_3 = Concatenate(axis=-1)([out_3, dout_2])

        deconv_3 = DeConv2D_stateful(filters=self.DeCNN_filter_list[2],
                                     keranel_size=(2, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     state_shape=self.DeCNN_state_list[2],
                                     name=name + '_dconv_3')(skipcon_3)
        dbn_3 = BatchNormalization(name=name + '_dbn_3')(deconv_3)
        dout_3 = PReLU(shared_axes=[1, 2])(dbn_3)

        skipcon_4 = Concatenate(axis=-1)([out_2, dout_3])

        deconv_4 = DeConv2D_stateful(filters=self.DeCNN_filter_list[3],
                                     keranel_size=(2, 3),
                                     strides=(1, 2),
                                     padding='same',
                                     state_shape=self.DeCNN_state_list[3],
                                     name=name + '_dconv_4')(skipcon_4)
        dbn_4 = BatchNormalization(name=name + '_dbn_4')(deconv_4)
        dout_4 = PReLU(shared_axes=[1, 2])(dbn_4)

        skipcon_5 = Concatenate(axis=-1)([out_1, dout_4])

        deconv_5 = DeConv2D_stateful(filters=self.DeCNN_filter_list[4],
                                     keranel_size=(2, 5),
                                     strides=(1, 2),
                                     padding='valid',
                                     state_shape=self.DeCNN_state_list[4],
                                     name=name + '_dconv_5')(skipcon_5)

        output_mask = deconv_5[:, 1:-1, :-2]
        enh_spec = Lambda(self.mk_mask)([real, imag, output_mask])
        # enh_real,  enh_imag = enh_spec[0], enh_spec[1]
        enh_frame = Lambda(self.ifftLayer, arguments={'mode': 'real_imag'})(enh_spec)[:, 0, :]
        enh_frame = enh_frame * win

        model = Model([Encoder_input], [enh_frame])
        model_encoder = Model(Encoder_input, dp_in)

        upop = tf.get_collection('upop')
        rsop = tf.get_collection('rsop')
        session_list = [[Encoder_input], [enh_frame, upop]]

        return model_encoder, model, session_list, rsop

    def test(self, filepath, model, sess, reset=False, reset_op=None, session_list=None):
        start_time = time.time()
        if reset:
            sess.run(reset_op)
        s = sf.read(filepath, dtype='float32')[0]
        audio_len = len(s)
        audio_length = (audio_len - blockLen) // blockshift + 1

        n_samples = (audio_length - 1) * blockshift + blockLen
        input_spec_data = np.array(s[:n_samples])
        input_spec_data = np.expand_dims(input_spec_data, axis=0)

        rt_output = []
        for i in range(audio_length):
            output_data = sess.run(session_list[1], feed_dict={model.input: input_spec_data[:, i * blockshift:i * blockshift + blockLen]})[0]
            rt_output.append(output_data)

        rt_output = np.concatenate(rt_output, axis=0)
        rt_output = self.overlapadd(rt_output)
        save_path = os.path.join(os.path.dirname(os.path.abspath(filepath)).replace("recording", "after_denoising"), "enhanced_" + os.path.basename(filepath))
        sf.write(save_path, rt_output.astype(np.float32), 16000)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"脚本运行时间: {execution_time} 秒")

        return rt_output

if __name__ == "__main__":
    start_time = time.time()
    # threadPool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio_denoise")    # 启动线程池
    mode = Noise_Suppression_Model()
    _, encoder, session_list, reset_op = mode.rnn_encoder()
    sess = backend.get_session()
    sess.run(tf.global_variables_initializer())

    encoder.load_weights('./train_mode/model_DPCRN_SNR+logMSE_causal_sinw.h5')
    filepath = glob.glob(os.path.join(r"./data/recording", "py*.wav"))
    for one_file in filepath:
        # mode.test(one_file, model=encoder, sess=sess, reset=True, reset_op=reset_op, session_list=session_list)
        # threadPool.submit(mode.test, one_file,model=encoder, sess=sess, reset=True, reset_op=reset_op, session_list=session_list)
        t1 = threading.Thread(target=mode.test, args=(one_file,encoder, sess, True, reset_op, session_list))
        t1.start()
        t1.join()
        # threadPool.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"脚本运行时间: {execution_time} 秒")
