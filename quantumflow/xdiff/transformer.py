# taken from https://www.tensorflow.org/text/tutorials/transformer

import tensorflow as tf
import numpy as np


def sin_cos(value, max_value, K, resolution=1, exponent=2, dampening=3, linspace=False, dtype=np.float32):
    value = tf.convert_to_tensor(value)[..., tf.newaxis] / max_value # (..., 1)
    k_bands = np.power(np.linspace(0, 1, K, dtype=dtype), exponent)
    
    k_bands = np.pi * (1 + k_bands * (max_value - 1.0 * linspace - resolution) / resolution)
    
    for _ in range(len(value.shape) - 1):
        k_bands = k_bands[np.newaxis] # (..., K)
    
    scaling = np.exp(-dampening * np.linspace(0, 1, K, dtype=dtype))
    
    k_cos = -tf.cos(value * k_bands) * scaling # (..., K)
    k_sin = tf.sin(value * k_bands) * scaling # (..., K)
    
    pos_encoding = np.sqrt(2) * tf.concat([k_cos, k_sin, -k_cos, -k_sin], axis=-1)
    
    if dampening > 0.0:
        pos_encoding /= tf.math.reduce_std(pos_encoding, axis=-1, keepdims=True)
        
    return tf.ensure_shape(pos_encoding, [None] * len(value.shape[:-1]) + [4 * K])


def get_xdiff(x1, x2, scale, K, **kwargs):
    xdiff = tf.expand_dims(x1, axis=-2) - tf.expand_dims(x2, axis=-3)
    xdiff = tf.sqrt(tf.reduce_sum(tf.square(xdiff), axis=-1)) # (..., x1_size, x2_size, 1)
    return sin_cos(xdiff, scale, K, **kwargs)
        

def scaled_dot_product_attention(q, k, v, initial_attention_logits=None, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: size_k = size_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., size_q, depth)
    k: key shape == (..., size_k, depth)
    v: value shape == (..., size_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., size_q, size_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., size_q, size_k)
        
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if initial_attention_logits is not None:
        scaled_attention_logits += initial_attention_logits
        
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
       
    # softmax is normalized on the last axis (size_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., size_q, size_k)
    
    output = tf.matmul(attention_weights, v)  # (..., size_q, depth_v)

    return output, attention_weights


class XdiffMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_attn, d_model, num_heads, num_x_features, kernel_initializer=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_attn = d_attn
        self.d_model = d_model
        self.num_x_features = num_x_features

        assert d_model % self.num_heads == 0

        self.depth = d_attn // self.num_heads

        self.wq = tf.keras.layers.Dense(d_attn, name='q')
        self.wk = tf.keras.layers.Dense(d_attn, name='k')
        self.wv = tf.keras.layers.Dense(d_attn, name='v', kernel_initializer=kernel_initializer)
        self.wx = tf.keras.layers.Dense(num_heads * num_x_features, name='x')
        
        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer, name='linear')

    def get_config(self):
        return {
            "d_attn": self.d_attn,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_x_features": self.num_x_features
        }
    
    def split_heads(self, x, batch_sizes):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (..., num_heads, size_x, depth)
        """
        x = tf.reshape(x, batch_sizes + [-1, self.num_heads, self.depth])
        slen = len(batch_sizes)
        return tf.transpose(x, perm=list(range(len(batch_sizes))) + [slen + 1, slen, slen + 2])

    def call(self, q, k, v, xdiff, mask=None):
        batch_sizes = tf.unstack(tf.shape(q)[:-2])

        q = self.wq(q) # (..., size_q, d_attn)
        k = self.wk(k) # (..., size_k, d_attn)
        v = self.wv(v) # (..., size_v, d_attn)
        x = self.wx(q) # (..., size_q, num_heads*num_x_features)
        
        q = self.split_heads(q, batch_sizes)  # (..., num_heads, size_q, depth)
        k = self.split_heads(k, batch_sizes)  # (..., num_heads, size_k, depth)
        v = self.split_heads(v, batch_sizes)  # (..., num_heads, size_v, depth)
        
        # xdiff (..., size_q, size_k, x_features)
        x = tf.reshape(x, batch_sizes + [-1, self.num_heads, self.num_x_features]) # (..., size_q, num_heads, num_x_features)
        x_diff_logits = tf.matmul(xdiff, x, transpose_b=True) # (..., size_q, size_k, num_heads)
        
        slen = len(batch_sizes)
        x_diff_logits = tf.transpose(x_diff_logits, perm=list(range(slen)) + [slen + 2, slen, slen + 1])
        
        dkx = tf.cast(tf.shape(xdiff)[-1], tf.float32)
        scaled_x_diff_logits = 2 * x_diff_logits / tf.math.sqrt(dkx)
        
        # scaled_attention.shape == (..., num_heads, size_q, depth)
        # attention_weights.shape == (..., num_heads, size_q, size_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, scaled_x_diff_logits, mask=mask)

        scaled_attention = tf.transpose(scaled_attention, perm=list(range(slen)) + [slen + 1, slen, slen + 2])
        # (..., size_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, batch_sizes + [-1, self.d_attn])  # (..., size_q, d_model)

        output = self.dense(concat_attention)  # (..., size_q, d_model)

        return output, attention_weights

    
class XdiffEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_x_features, dff, activation='gelu', dropout_rate=0.1, kernel_initializer=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_x_features = num_x_features
        self.dff = dff

        self.mha = XdiffMultiHeadAttention(d_model, d_model, num_heads, num_x_features, kernel_initializer=kernel_initializer)
        
        self.ffn = [
            tf.keras.layers.Dense(dff, activation=activation, kernel_initializer=kernel_initializer),  # (..., seq_len, dff)
            tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)  # (..., seq_len, d_model)
        ]

        self.layernorm1 = tf.keras.layers.LayerNormalization(name='latents_layernorm')
        self.layernorm2 = tf.keras.layers.LayerNormalization(name='ffn_layernorm')

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_x_features": self.num_x_features,
            "dff": self.dff,
        }
    
    def call(self, latents, xdiff, training=False, mask=None):
        lat = self.layernorm1(latents)  # (..., input_size, d_model)
        
        if self.debug and tf.executing_eagerly():
            for i in range(lat.shape[2]):
                plt.figure(figsize=(20, 3))
                plt.plot(lat[0, :, i, :])
                plt.title(f"Normalized Latents")
                plt.show()
                
                
        lat = tf.nn.gelu(lat)
        attn_output, _ = self.mha(lat, lat, lat, xdiff, mask=mask)  # (..., input_size, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        
        latents = latents + attn_output
        
        lat = self.layernorm2(latents)
        lat = tf.nn.gelu(lat)
        ffn_output = self.ffn[1](self.ffn[0](lat))  # (..., input_size, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        return latents + ffn_output  # (..., input_size, d_model)

    
class XdiffCrossEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_cross, d_model, num_heads, num_x_features, dff, activation='gelu', dropout_rate=0.1, kernel_initializer=None):
        super().__init__()
        
        self.d_cross = d_cross
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_x_features = num_x_features
        self.dff = dff
        
        self.mha = XdiffMultiHeadAttention(d_cross, d_model, num_heads, num_x_features, kernel_initializer=kernel_initializer)
        
        self.ffn = [
            tf.keras.layers.Dense(dff, activation=activation, kernel_initializer=kernel_initializer, name='ffn_0'),  # (..., seq_len, dff)
            tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer, name='ffn_1')  # (..., seq_len, d_model)
        ]
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(name='latents_layernorm')
        self.layernorm2 = tf.keras.layers.LayerNormalization(name='ffn_layernorm')

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
    def get_config(self):
        return {
            "d_cross": self.d_cross,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_x_features": self.num_x_features,
            "dff": self.dff,
        }

    def call(self, latents, inputs, xdiff_cross, training=False, mask=None):
        inp = inputs
        inp = tf.nn.gelu(inputs)
        lat = self.layernorm1(latents)
        lat = tf.nn.gelu(lat)
        
        attn_output, _ = self.mha(lat, inp, inp, xdiff_cross, mask=mask)  # (..., latent_size, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        
        latents = latents + attn_output

        lat = self.layernorm2(latents)
        lat = tf.nn.gelu(lat)
        ffn_output = self.ffn[1](self.ffn[0](lat))  # (..., input_size, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        return latents + ffn_output  # (..., latent_size, d_model)
        
    
class XdiffPerciever(tf.keras.layers.Layer):
    def __init__(self, num_outputs, num_layers, num_repeats, d_cross, d_model, num_heads, dff, dff_final, share_weights=False,
                 activation='gelu', kernel_scale=None, dropout_rate=0.1, dampening=3, exponent=2,
                 K_xdiff=10, scale_xdiff=1.0, K_input=10, scale_input=1.0, res_xdiff=0.01, res_input=0.01):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_repeats = num_repeats
        self.d_cross = d_cross
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.dff = dff
        self.dff_final = dff_final
        
        self.share_weights = share_weights
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        self.exponent = exponent
        self.dampening = dampening
        
        self.K_xdiff = K_xdiff
        self.scale_xdiff = scale_xdiff
        self.res_xdiff = res_xdiff
        self.K_input = K_input
        self.scale_input = scale_input
        self.res_input = res_input
        
        if kernel_scale is None:
            kernel_scale = float(1 / (num_layers * num_repeats + num_layers + num_repeats))
        
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=kernel_scale, mode='fan_avg', distribution='uniform') # scaled Glorot uniform

        self.x_token = self.add_weight(name='x_token', shape=(1, d_model), dtype=tf.float32, trainable=True,
                                       initializer=tf.keras.initializers.RandomNormal(stddev=kernel_scale)) # (d_model)
        
        num_x_features = 4 * K_xdiff
        self.enc_layers = [[XdiffEncoderLayer(d_model, num_heads, num_x_features, dff, activation=activation,
                                              dropout_rate=dropout_rate, kernel_initializer=self.kernel_initializer) for _ in range(num_layers)] for _ in range(num_repeats)]
        self.cross_enc_layers = [XdiffCrossEncoderLayer(d_cross, d_model, num_heads, num_x_features, dff, activation=activation,
                                                        dropout_rate=dropout_rate, kernel_initializer=self.kernel_initializer) for _ in range(num_repeats)]

        self.layernorm = tf.keras.layers.LayerNormalization(name='final_layernorm')
        
        self.pre_final_layers = [tf.keras.layers.Dense(dff, activation=activation, name=f'output_ffn_{d}') for d, dff in enumerate(dff_final)]
        self.final_layer = tf.keras.layers.Dense(num_outputs, name=f'output_dense_layer')
        
        self.debug = False
    
    def get_config(self):
        return {
            "num_outputs": self.num_outputs,
            "num_layers": self.num_layers,
            "num_repeats": self.num_repeats,
            "d_cross": self.d_cross,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dff_final": self.dff_final,
            "dropout_rate": self.dropout_rate,
            "scale": self.scale,
        }
    
    def call(self, x, x_inputs, inputs, training=False, mask=None):
        x_token = self.x_token # (d_model)
        for shape in tf.unstack(tf.shape(x))[:-2]:
            x_token = tf.repeat(tf.expand_dims(x_token, axis=-3), shape, axis=-3) # (..., latent_size, d_model)
        x_token = tf.repeat(x_token, tf.shape(x)[-2], axis=0)
        
        xdiff = get_xdiff(x, x, self.scale_xdiff, self.K_xdiff, resolution=self.res_xdiff, exponent=self.exponent, dampening=self.dampening) # (..., latent_size, latent_size, x_features)
        xdiff_cross = get_xdiff(x, x_inputs, self.scale_xdiff, self.K_xdiff, resolution=self.res_xdiff, exponent=self.exponent, dampening=self.dampening) # (..., latent_size, input_size, x_features)
        
        latents = x_token # self.x_token_layer(xdiff, xdiff_cross)
        
        if self.debug and tf.executing_eagerly():
            import matplotlib.pyplot as plt
            for i in range(latents.shape[2]):
                plt.figure(figsize=(20, 3))
                plt.plot(latents.numpy()[0, :, i, :])
                plt.title(f"Latents {np.mean(latents.numpy()[0, :, i, :]):.3f} {np.std(latents.numpy()[0, :, i, :]):.3f}")
                plt.show()
                
        inputs = sin_cos(inputs, self.scale_input, self.K_input, resolution=self.res_input, exponent=self.exponent, dampening=self.dampening) # (..., x1_size, x2_size, x_features)
        
        for r in range(self.num_repeats):
            latents = self.cross_enc_layers[r](latents, inputs, xdiff_cross, training=training, mask=mask)
            
            for i in range(self.num_layers):
                latents = self.enc_layers[r][i](latents, xdiff, training=training, mask=None)
            
        latents = self.layernorm(latents)
        
        for layer in self.pre_final_layers:
            latents = layer(latents)
        
        outputs = self.final_layer(latents)
        
        return outputs # (..., latent_size, num_outputs)
