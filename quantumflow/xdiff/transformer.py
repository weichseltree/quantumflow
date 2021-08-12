# taken from https://www.tensorflow.org/text/tutorials/transformer

import tensorflow as tf
import numpy as np


def get_xdiff(x1, x2):
    xdiff = tf.expand_dims(x1, axis=-2) - tf.expand_dims(x2, axis=-3)
    return tf.sqrt(tf.reduce_sum(tf.square(xdiff), axis=-1)) # (..., seq_len, seq_len)

def point_wise_feed_forward_network(d_model, dff, activation='relu'):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=activation),  # (..., seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (..., seq_len, d_model)
    ])

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
    
    if initial_attention_logits is not None:
        matmul_qk += initial_attention_logits
        
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
       
    # softmax is normalized on the last axis (size_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., size_q, size_k)
    
    output = tf.matmul(attention_weights, v)  # (..., size_q, depth_v)

    return output, attention_weights


class XdiffMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, name='q')
        self.wk = tf.keras.layers.Dense(d_model, name='k')
        self.wv = tf.keras.layers.Dense(d_model, name='v')
        
        self.wa = tf.keras.layers.Dense(self.num_heads, name='a')
        self.wb = tf.keras.layers.Dense(self.num_heads, name='b')
        self.wc = tf.keras.layers.Dense(self.num_heads, name='c')
        self.wd = tf.keras.layers.Dense(self.num_heads, name='d')
        self.we = tf.keras.layers.Dense(self.num_heads, name='e')
        self.wf = tf.keras.layers.Dense(self.num_heads, name='f')
        self.wg = tf.keras.layers.Dense(self.num_heads, name='g')
        self.wh = tf.keras.layers.Dense(self.num_heads, name='h')
        self.wi = tf.keras.layers.Dense(self.num_heads, name='i')
        self.wj = tf.keras.layers.Dense(self.num_heads, name='j')
        
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_sizes):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (..., num_heads, size_x, depth)
        """
        x = tf.reshape(x, batch_sizes + [-1, self.num_heads, self.depth])
        return tf.transpose(x, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes), len(batch_sizes)+2])

    def call(self, q, k, v, xdiff, mask=None):
        batch_sizes = tf.unstack(tf.shape(q)[:-2])

        q = self.wq(q)  # (..., size_q, d_model)
        k = self.wk(k)  # (..., size_k, d_model)
        v = self.wv(v)  # (..., size_v, d_model)
        
        a = self.wa(q)  # (..., size_q, num_heads)
        b = self.wb(q)  # (..., size_q, num_heads)
        c = self.wc(q)  # (..., size_q, num_heads)
        d = self.wd(q)  # (..., size_q, num_heads)
        e = self.we(q)  # (..., size_q, num_heads)
        f = self.wf(q)  # (..., size_q, num_heads)
        g = self.wg(q)  # (..., size_q, num_heads)
        h = self.wh(q)  # (..., size_q, num_heads)
        i = self.wi(q)  # (..., size_q, num_heads)
        j = self.wj(q)  # (..., size_q, num_heads)
        
        q = self.split_heads(q, batch_sizes)  # (..., num_heads, size_q, depth)
        k = self.split_heads(k, batch_sizes)  # (..., num_heads, size_k, depth)
        v = self.split_heads(v, batch_sizes)  # (..., num_heads, size_v, depth)
        
        a = tf.transpose(a, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        b = tf.transpose(b, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        c = tf.transpose(c, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        d = tf.transpose(d, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        e = tf.transpose(e, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        f = tf.transpose(f, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        g = tf.transpose(g, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        h = tf.transpose(h, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        i = tf.transpose(i, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        j = tf.transpose(j, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes)])  # (..., num_heads, size_q)
        
        xdiff = tf.expand_dims(xdiff, axis=-3) # (..., num_heads, size_q, size_k)
        
        initial_attention_logits = 0.1*(tf.expand_dims(a, axis=-1)*xdiff \
                                   - 0.5*tf.abs(tf.expand_dims(b, axis=-1)) * xdiff**2 \
                                   + tf.expand_dims(c, axis=-1) * tf.sin(np.pi*xdiff) \
                                   + tf.expand_dims(d, axis=-1) * tf.cos(np.pi*xdiff) \
                                   + tf.expand_dims(e, axis=-1) * tf.sin(np.pi*3*xdiff) \
                                   + tf.expand_dims(f, axis=-1) * tf.cos(np.pi*3*xdiff) \
                                   + tf.expand_dims(g, axis=-1) * tf.sin(np.pi*5*xdiff) \
                                   + tf.expand_dims(h, axis=-1) * tf.cos(np.pi*5*xdiff) \
                                   + tf.expand_dims(i, axis=-1) * tf.sin(np.pi*9*xdiff) \
                                   + tf.expand_dims(j, axis=-1) * tf.cos(np.pi*9*xdiff))
         
        # scaled_attention.shape == (..., num_heads, size_q, depth)
        # attention_weights.shape == (..., num_heads, size_q, size_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, initial_attention_logits, mask=mask)

        scaled_attention = tf.transpose(scaled_attention, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes), len(batch_sizes)+2])  
        # (..., size_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, batch_sizes + [-1, self.d_model])  # (..., size_q, d_model)

        output = self.dense(concat_attention)  # (..., size_q, d_model)

        return output, attention_weights

    
class XdiffEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, activation='relu', dropout_rate=0.1):
        super().__init__()

        self.mha = XdiffMultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff, activation=activation)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, xdiff, training=False, mask=None):
        inp = self.layernorm1(inputs)  # (..., input_size, d_model)

        attn_output, _ = self.mha(inp, inp, inp, xdiff, mask=mask)  # (..., input_size, d_model)
        
        attn_output = self.dropout1(attn_output, training=training)
        #out1 = self.layernorm1(inputs + attn_output)  # (..., input_size, d_model)
        out1 = inputs + attn_output

        ffn_output = self.ffn(out1)  # (..., input_size, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (..., input_size, d_model)

        return out2

    
class XdiffTransformer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, num_layers, d_model, num_heads, dff_input, dff, dff_final, activation='relu', dropout_rate=0.1, scale=1.0):
        super().__init__()
        self.num_outputs = num_outputs
        self.d_model = d_model
        self.num_layers = num_layers
        self.dff_input = dff_input
        self.dff_final = dff_final
        self.dff = dff
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        self.scale = scale

        self.input_layers = [tf.keras.layers.Dense(d_model, activation='softplus' if d == 0 else activation) for d, dff in enumerate(dff_input)]
        self.x_token = self.add_weight(name='x_token', shape=(d_model,), dtype=tf.float32, trainable=True) # (d_model)
    
        self.enc_layers = [XdiffEncoderLayer(d_model, num_heads, dff, activation=activation, dropout_rate=dropout_rate) for _ in range(num_layers)]

        self.pre_final_layers = [tf.keras.layers.Dense(dff, activation=activation) for dff in dff_final]
        self.final_layer = tf.keras.layers.Dense(num_outputs)
    
    def get_config(self):
        return {
            "num_outputs": self.num_outputs,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff_input": self.dff_input,
            "activation": self.activation,
            "dff": self.dff,
            "dff_final": self.dff_final,
            "dropout_rate": self.dropout_rate,
            "scale": self.scale,
        }
    
    def call(self, x, x_inputs, inputs, training=False, mask=None):        
        x_all = tf.concat([tf.expand_dims(x, axis=-2), x_inputs], axis=-2) # (..., input_size+1, num_dims)
        xdiff = get_xdiff(x_all, x_all)/self.scale # (..., input_size+1, input_size+1)
        
        x_token = self.x_token # (d_model)
        for shape in tf.unstack(tf.shape(x))[:-1]:
            x_token = tf.repeat(tf.expand_dims(x_token, axis=-2), shape, axis=-2)
        x_token = tf.expand_dims(x_token, axis=-2) # (..., 1, d_model)
        
        value = inputs
        for layer in self.input_layers:
            value = layer(value)
            
        value = tf.concat([x_token, value], axis=-2)
        
        for i in range(self.num_layers):
            value = self.enc_layers[i](value, xdiff, training=training, mask=mask)

        value = value[..., 0, :]
        
        for layer in self.pre_final_layers:
            value = layer(value)
        
        outputs = self.final_layer(value)
        
        return outputs # (..., num_outputs)


class TFWhileXdiffTransformer(XdiffTransformer):
    
    def call(self, all_x, all_x_inputs, all_inputs, training=False, mask=None):
        num_x = tf.shape(all_x)[1]

        x = tf.TensorArray(all_x.dtype, num_x).unstack(tf.transpose(all_x, perm=[1, 0, 2]))
        x_inputs = tf.TensorArray(all_x_inputs.dtype, num_x).unstack(tf.transpose(all_x_inputs, perm=[1, 0, 2, 3]))
        inputs = tf.TensorArray(all_inputs.dtype, num_x).unstack(tf.transpose(all_inputs, perm=[1, 0, 2, 3]))
        
        output = super().call(x.read(0), x_inputs.read(0), inputs.read(0), training=training, mask=mask)

        output_x = tf.TensorArray(output.dtype, num_x).write(0, output)

        for t in tf.range(1, num_x):
            tf.autograph.experimental.set_loop_options(
                parallel_iterations=1,
                swap_memory=True,
                maximum_iterations=num_x-1)

            output = super().call(x.read(t), x_inputs.read(t), inputs.read(t), training=training, mask=mask)
            output_x = output_x.write(t, output)

        '''
        output = tf.TensorArray(inputs.dtype, num_x)

        def step(i, x, x_inputs, inputs, output):
            output = output.write(i, self.crazynet(x.read(i), x_inputs.read(i), inputs.read(i), training=training, mask=mask))
            return i + 1, x, x_inputs, inputs, output
        
        _, _, _, _, output = tf.while_loop(
            cond=lambda i, x, x_inputs, inputs, output: tf.less(i, num_x),
            body=step,
            loop_vars = (
                tf.constant(0, dtype=tf.int32),
                x, x_inputs, inputs, output
            ),
            parallel_iterations=1,
            swap_memory=True,
            maximum_iterations=num_x-1
        )
        '''
            
        all_output = tf.transpose(tf.TensorArray.stack(output_x), perm=[1, 0, 2])
        return all_output


##################################################################################################################################

    
class XdiffCrossEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, activation='relu', dropout_rate=0.1):
        super().__init__()

        self.mha = XdiffMultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff, activation=activation)

        self.layernorm1b = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm1a = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, latents, inputs, xdiff_cross, training=False, mask=None):
        
        lat = self.layernorm1a(latents)
        inp = inputs #self.layernorm1b(inputs)
        
        attn_output, _ = self.mha(lat, inp, inp, xdiff_cross, mask=mask)  # (..., latent_size, d_model)

        attn_output = self.dropout1(attn_output, training=training)
        #out1 = self.layernorm1(latents + attn_output)  # (..., latent_size, d_model)
        out1 = latents + self.layernorm1b(attn_output)

        ffn_output = self.ffn(out1)  # (..., latent_size, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (..., latent_size, d_model)

        return out2
    
    
class XdiffPerciever(tf.keras.layers.Layer):
    def __init__(self, num_outputs, num_layers, num_repeats, latents_per_x, d_model, num_heads, dff_input, dff, dff_final, share_weights=False, activation='relu', dropout_rate=0.1, scale=1.0):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_repeats = num_repeats
        self.d_model = d_model
        self.num_heads = num_heads
        self.latents_per_x = latents_per_x
        
        self.dff_input = dff_input
        self.dff = dff
        self.dff_final = dff_final
        
        self.share_weights = share_weights
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        self.scale = scale

        self.input_layers = [tf.keras.layers.Dense(d_model, activation='softplus' if d == 0 else activation) for d, dff in enumerate(dff_input)]
        self.x_token = self.add_weight(name='x_token', shape=(latents_per_x, d_model), dtype=tf.float32, trainable=True) # (d_model)
        
        self.enc_layers = [[XdiffEncoderLayer(d_model, num_heads, dff, activation=activation, dropout_rate=dropout_rate) for _ in range(num_layers)] for _ in range(num_repeats+1)]
        self.cross_enc_layers = [XdiffCrossEncoderLayer(d_model, num_heads, dff, activation=activation, dropout_rate=dropout_rate) for _ in range(num_repeats)]

        self.pre_final_layers = [tf.keras.layers.Dense(dff, activation=activation) for dff in dff_final]
        self.final_layer = tf.keras.layers.Dense(num_outputs)
    
    def get_config(self):
        return {
            "num_outputs": self.num_outputs,
            "num_layers": self.num_layers,
            "num_repeats": self.num_repeats,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_repeats": self.num_repeats,
            "latents_per_x": self.latents_per_x,
            "dff_input": self.dff_input,
            "dff": self.dff,
            "dff_final": self.dff_final,
            "dropout_rate": self.dropout_rate,
            "scale": self.scale,
        }
    
    def call(self, x_outputs, x_inputs, inputs, training=False, mask=None):    
        x_token = self.x_token # (d_model)
        for shape in tf.unstack(tf.shape(x_outputs))[:-2]:
            x_token = tf.repeat(tf.expand_dims(x_token, axis=-3), shape, axis=-3) # (..., latent_size, d_model)
        x_token = tf.repeat(x_token, tf.shape(x_outputs)[-2], axis=0)
        
        x_outputs = tf.repeat(x_outputs, self.latents_per_x, axis=-2)
        xdiff = get_xdiff(x_outputs, x_outputs)/self.scale # (..., latent_size, latent_size)
        xdiff_cross = get_xdiff(x_outputs, x_inputs)/self.scale # (..., latent_size, input_size)
        
        for layer in self.input_layers:
            inputs = layer(inputs)
            
        latents = x_token
        
        for r in range(self.num_repeats):
            for i in range(self.num_layers):
                latents = self.enc_layers[r][i](latents, xdiff, training=training, mask=None)
            latents = self.cross_enc_layers[r](latents, inputs, xdiff_cross, training=training, mask=mask)

        for i in range(self.num_layers):
            latents = self.enc_layers[self.num_repeats][i](latents, xdiff, training=training, mask=None)
            
        for layer in self.pre_final_layers:
            latents = layer(latents)
        
        outputs = self.final_layer(latents)
        
        return outputs # (..., latent_size, num_outputs)

    def get_core(self, x_outputs, x_inputs, inputs, training=False, mask=None):
        x_token = self.x_token # (d_model)
        for shape in tf.unstack(tf.shape(x_outputs))[:-2]:
            x_token = tf.repeat(tf.expand_dims(x_token, axis=-3), shape, axis=-3) # (..., latent_size, d_model)
        x_token = tf.repeat(x_token, tf.shape(x_outputs)[-2], axis=0)
        
        x_outputs = tf.repeat(x_outputs, self.latents_per_x, axis=-2)
        xdiff = get_xdiff(x_outputs, x_outputs)/self.scale # (..., latent_size, latent_size)
        xdiff_cross = get_xdiff(x_outputs, x_inputs)/self.scale # (..., latent_size, input_size)
        
        for layer in self.input_layers:
            inputs = layer(inputs)
            
        latents = x_token
        
        layers = []
        for r in range(self.num_repeats):
            for i in range(self.num_layers):
                layers.append(self.enc_layers[r][i])
            layers.append(self.cross_enc_layers[r])

        for i in range(self.num_layers):
            layers.append(self.enc_layers[self.num_repeats][i])
            
        for layer in self.pre_final_layers:
            layers.append(layer)
            
        layers.append(self.final_layer)
        
        return latents, xdiff, inputs, xdiff_cross, layers
