import tensorflow as tf
from tensorflow.keras.layers import Dense, TimeDistributed, GlobalAveragePooling1D, Flatten, Permute, Dropout, LayerNormalization, Conv2D
import tensorflow.keras.backend as K
from einops import rearrange
import numpy as np

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def mlp_layer(x, hidden_dim, output_dim):
    x = Dense(hidden_dim, activation=gelu)(x)
    output = Dense(output_dim)(x)
    return output

def mixer_layer(x, token_h, channel_h):
    # layer norm
    y = LayerNormalization()(x)
    y = Dropout(0.2)(y)
    
    # token mixing
    y = Permute((2, 1))(y)
    y = mlp_layer(y, token_h, y.shape[-1])
    print('1st mlp_layer :', y.shape)
    y = Permute((2, 1))(y)
    # skip connection
    x = y + x
    
    # layer norm
    y = LayerNormalization()(x)
    y = Dropout(0.2)(y)
    
    # channel mixing
    y = mlp_layer(y, channel_h, y.shape[-1])
    print('2nd mlp_layer :', y.shape)
    # skip connection
    return x + y
    
def patch_layer(x, patch_size, output_dim):
    x = tf.keras.layers.experimental.preprocessing.RandomFlip()(x)
    x = Conv2D(output_dim, (patch_size, patch_size), strides=(patch_size, patch_size), padding='valid')(x)
    x = Permute((3, 1, 2))(x)
    print('permute :', x.shape)

    #     x = rearrange(x, 'b h w c -> b (h w) c')
    x = TimeDistributed(Flatten())(x)
    x = Permute((2, 1))(x)
    return x
# B/16 model
def mixer_model(input_dim, output_dim, patch_size=16, c_dim=int(768/4), dc_dim=int(3072/4), ds_dim=int(384/4), layer_num=12):
    inputs = tf.keras.Input(shape=input_dim)
    x = patch_layer(inputs, patch_size, c_dim)
    for i in range(layer_num):
        x = mixer_layer(x, dc_dim, ds_dim)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)#data_format='channels_first')(x)
    print('gap :', x.shape)
    outputs = Dense(output_dim, activation='softmax')(x)    
    return tf.keras.Model(inputs, outputs, name='mixer_model')


    