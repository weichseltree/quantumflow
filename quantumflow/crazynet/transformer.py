# taken from https://www.tensorflow.org/text/tutorials/transformer

import tensorflow as tf
import numpy as np


def get_xdiff(x1, x2):
    xdiff = tf.expand_dims(x1, axis=-2) - tf.expand_dims(x2, axis=-3)
    return tf.sqrt(tf.reduce_sum(tf.square(xdiff), axis=-1)) # (..., seq_len, seq_len)

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (..., seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (..., seq_len, d_model)
    ])

def metric_scaled_dot_product_attention(q, k, v, alpha, beta, xdiff, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: size_k = size_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    alpha: linear parameter for metric == (..., size_q)
    beta: quadratic parameter for metric == (..., size_q)
    xdiff: distances of points from metric == (..., size_q, size_k)
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

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        
    # v1
    metric_attention_logits = tf.expand_dims(alpha, axis=-1)*xdiff + tf.expand_dims(beta, axis=-1) * xdiff**2
    
    # v2
    #beta = tf.sqrt(tf.sqrt(tf.square(beta)+1))-1
    #metric_attention_logits = -0.5*tf.expand_dims(beta, axis=-1) * tf.square(xdiff - tf.expand_dims(alpha, axis=-1))
    
    # softmax is normalized on the last axis (size_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits + metric_attention_logits, axis=-1)  # (..., size_q, size_k)
    
    output = tf.matmul(attention_weights, v)  # (..., size_q, depth_v)

    return output, attention_weights


class MetricMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.wa = tf.keras.layers.Dense(self.num_heads)#, kernel_initializer='zeros')
        self.wb = tf.keras.layers.Dense(self.num_heads)#, kernel_initializer='zeros')

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_sizes):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (..., num_heads, size_x, depth)
        """
        x = tf.reshape(x, batch_sizes + [-1, self.num_heads, self.depth])
        return tf.transpose(x, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes), len(batch_sizes)+2])

    def call(self, v, k, q, xdiff, mask=None):
        batch_sizes = tf.unstack(tf.shape(q)[:-2])

        q = self.wq(q)  # (..., size_q, d_model)
        k = self.wk(k)  # (..., size_k, d_model)
        v = self.wv(v)  # (..., size_v, d_model)
        a = self.wa(q)  # (..., size_q, num_heads)
        b = self.wb(q)  # (..., size_q, num_heads)
        
        
        q = self.split_heads(q, batch_sizes)  # (..., num_heads, size_q, depth)
        k = self.split_heads(k, batch_sizes)  # (..., num_heads, size_k, depth)
        v = self.split_heads(v, batch_sizes)  # (..., num_heads, size_v, depth)
        
        a = tf.transpose(a, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        b = tf.transpose(b, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        
        xdiff = tf.expand_dims(xdiff, axis=-3) # (..., num_heads, size_q, size_k)
        
        # scaled_attention.shape == (batch_size, num_heads, size_q, depth)
        # attention_weights.shape == (batch_size, num_heads, size_q, size_k)
        scaled_attention, attention_weights = metric_scaled_dot_product_attention(q, k, v, a, b, xdiff, mask=mask)

        scaled_attention = tf.transpose(scaled_attention, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes), len(batch_sizes)+2])  # (batch_size, size_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, batch_sizes + [-1, self.d_model])  # (batch_size, size_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, size_q, d_model)

        return output, attention_weights

    
class MetricEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.mha = MetricMultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, xdiff, training=False, mask=None):

        attn_output, _ = self.mha(inputs, inputs, inputs, xdiff, mask=mask)  # (..., input_size, d_model)
        
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # (..., input_size, d_model)

        ffn_output = self.ffn(out1)  # (..., input_size, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (..., input_size, d_model)

        return out2

    
class CrazyNet(tf.keras.layers.Layer):
    def __init__(self, num_outputs, num_layers, d_model, num_heads, dff, dff_final, dropout_rate=0.1, scale=1.0):
        super().__init__()
        self.num_outputs = num_outputs
        self.d_model = d_model
        self.num_layers = num_layers
        self.dff_final = dff_final
        
        self.scale = scale

        self.input_layer = tf.keras.layers.Dense(d_model, activation='relu')
        self.x_token = self.add_weight(name='x_token', shape=(d_model,), dtype=tf.float32, trainable=True) # (d_model)
    
        self.enc_layers = [MetricEncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.pre_final_layers = [tf.keras.layers.Dense(dff, activation='relu') for dff in dff_final]
        self.final_layer = tf.keras.layers.Dense(num_outputs)#, kernel_initializer='zeros')
    
    
    def call(self, x, x_inputs, inputs, training=False, mask=None):
        x_all = tf.concat([tf.expand_dims(x, axis=-2), x_inputs], axis=-2) # (..., input_size+1, num_dims)
        xdiff = get_xdiff(x_all, x_all)/self.scale # (..., input_size+1, input_size+1)
        
        x_token = self.x_token # (d_model)
        for shape in tf.unstack(tf.shape(x))[:-1]:
            x_token = tf.repeat(tf.expand_dims(x_token, axis=-2), shape, axis=-2)
        x_token = tf.expand_dims(x_token, axis=-2) # (..., 1, d_model)
        
        value = all_inputs = tf.concat([x_token, self.input_layer(inputs)], axis=-2)
        
        for i in range(self.num_layers):
            value = self.enc_layers[i](value, xdiff, training=training, mask=mask)

        value = value[..., 0]
        
        for layer in self.pre_final_layers:
            value = layer(value)
        
        outputs = self.final_layer(value)
        
        return outputs # (..., num_outputs)
    
    