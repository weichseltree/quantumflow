# taken from https://www.tensorflow.org/text/tutorials/transformer

import tensorflow as tf
import numpy as np


def get_xdiff(x1, x2):
    xdiff = tf.expand_dims(x1, axis=-2) - tf.expand_dims(x2, axis=-3)
    xdiff =  tf.sqrt(tf.reduce_sum(tf.square(xdiff), axis=-1)) # (..., seq_len, seq_len)
    
    xdiff = tf.stack([
        xdiff, 
        xdiff**2,
        tf.sin(np.pi*xdiff),
        tf.cos(np.pi*xdiff),
        tf.sin(np.pi*2*xdiff),
        tf.cos(np.pi*2*xdiff),
        tf.sin(np.pi*3*xdiff),
        tf.cos(np.pi*3*xdiff),
        tf.sin(np.pi*4*xdiff),
        tf.cos(np.pi*4*xdiff),
        tf.sin(np.pi*5*xdiff),
        tf.cos(np.pi*5*xdiff),
        tf.sin(np.pi*6*xdiff),
        tf.cos(np.pi*6*xdiff),
        tf.sin(np.pi*7*xdiff),
        tf.cos(np.pi*7*xdiff),
        tf.sin(np.pi*8*xdiff),
        tf.cos(np.pi*8*xdiff),
        tf.sin(np.pi*9*xdiff),
        tf.cos(np.pi*9*xdiff),
        tf.sin(np.pi*10*xdiff),
        tf.cos(np.pi*10*xdiff),
        tf.sin(np.pi*11*xdiff),
        tf.cos(np.pi*11*xdiff),
        tf.sin(np.pi*12*xdiff),
        tf.cos(np.pi*12*xdiff),
    ], axis=-1) # (..., latent_size, latent_size, x_features)
    return xdiff
        


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
        self.wx = tf.keras.layers.Dense(num_heads*num_x_features, name='x')
        
        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer, name='linear')

    def split_heads(self, x, batch_sizes):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (..., num_heads, size_x, depth)
        """
        x = tf.reshape(x, batch_sizes + [-1, self.num_heads, self.depth])
        return tf.transpose(x, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes), len(batch_sizes)+2])

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
        
        x_diff_logits = tf.transpose(x_diff_logits, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+2, len(batch_sizes), len(batch_sizes)+1])  
        
        dkx = tf.cast(tf.shape(xdiff)[-1], tf.float32)
        scaled_x_diff_logits = x_diff_logits / tf.math.sqrt(dkx)
        
        # scaled_attention.shape == (..., num_heads, size_q, depth)
        # attention_weights.shape == (..., num_heads, size_q, size_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, scaled_x_diff_logits, mask=mask)

        scaled_attention = tf.transpose(scaled_attention, perm=list(range(len(batch_sizes))) + [len(batch_sizes)+1, len(batch_sizes), len(batch_sizes)+2])  
        # (..., size_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, batch_sizes + [-1, self.d_attn])  # (..., size_q, d_model)

        output = self.dense(concat_attention)  # (..., size_q, d_model)

        return output, attention_weights

    
class XdiffEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, num_x_features, dff, activation='gelu', dropout_rate=0.1, kernel_initializer=None):
        super().__init__()

        self.mha = XdiffMultiHeadAttention(d_model, d_model, num_heads, num_x_features, kernel_initializer=kernel_initializer)
        
        self.ffn = [
            tf.keras.layers.Dense(dff, activation=activation, kernel_initializer=kernel_initializer),  # (..., seq_len, dff)
            tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer)  # (..., seq_len, d_model)
        ]

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='latents_layernorm')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ffn_layernorm')

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, latents, xdiff, training=False, mask=None):
        lat = self.layernorm1(latents)  # (..., input_size, d_model)
        attn_output, _ = self.mha(lat, lat, lat, xdiff, mask=mask)  # (..., input_size, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        
        latents = latents + attn_output
        
        lat = self.layernorm2(latents)
        ffn_output = self.ffn[1](self.ffn[0](lat))  # (..., input_size, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        return latents + ffn_output  # (..., input_size, d_model)

    
class XdiffTransformer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, num_layers, d_model, num_heads, dff_input, dff, dff_final, 
                 activation='gelu', kernel_scale=None, dropout_rate=0.1, scale=1.0):
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
        
        if kernel_scale is None:
            kernel_scale = 1/(2*num_layers)
            
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=kernel_scale, mode='fan_avg', distribution='uniform') # scaled Glorot uniform

        self.input_layers = [tf.keras.layers.Dense(d_model, activation='softplus' if d == 0 else activation) for d, dff in enumerate(dff_input)]
        self.x_token = self.add_weight(name='x_token', shape=(d_model,), dtype=tf.float32, trainable=True, 
                                       initializer=tf.keras.initializers.RandomNormal(stddev=kernel_scale)) # (d_model)
    
        self.enc_layers = [XdiffEncoderLayer(d_model, num_heads, dff, activation=activation, 
                                             dropout_rate=dropout_rate, kernel_initializer=self.kernel_initializer) for _ in range(num_layers)]

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='final_layernorm')
        
        self.pre_final_layers = [tf.keras.layers.Dense(dff, activation=activation, name=f'output_ffn_{d}') for d, dff in enumerate(dff_final)]
        self.final_layer = tf.keras.layers.Dense(num_outputs, name=f'output_dense_layer')
    
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
        
        latents = inputs
        for layer in self.input_layers:
            latents = layer(latents)
            
        latents = tf.concat([x_token, latents], axis=-2)
        
        for i in range(self.num_layers):
            latents = self.enc_layers[i](latents, xdiff, training=training, mask=mask)
        
        latents = self.layernorm(latents)
        latents = latents[..., 0, :]
        
        for layer in self.pre_final_layers:
            latents = layer(latents)
        
        outputs = self.final_layer(latents)
        
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
    def __init__(self, d_cross, d_model, num_heads, num_x_features, dff, dff_input, activation='gelu', dropout_rate=0.1, kernel_initializer=None):
        super().__init__()
        
        self.input_layers = [tf.keras.layers.Dense(dff_i, activation=activation, name=f'inputs_ffn_{d}') for d, dff_i in enumerate(dff_input)]

        self.mha = XdiffMultiHeadAttention(d_cross, d_model, num_heads, num_x_features, kernel_initializer=kernel_initializer)
        
        self.ffn = [
            tf.keras.layers.Dense(dff, activation=activation, kernel_initializer=kernel_initializer, name='ffn_0'),  # (..., seq_len, dff)
            tf.keras.layers.Dense(d_model, kernel_initializer=kernel_initializer, name='ffn_1')  # (..., seq_len, d_model)
        ]
        
        self.layernorm1a = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='latents_layernorm')
        self.layernorm1b = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='inputs_layernorm')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ffn_layernorm')

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, latents, inputs, xdiff_cross, training=False, mask=None):
        
        inp = inputs
        for layer in self.input_layers:
            inp = layer(inp)
        
        lat = self.layernorm1a(latents)
        inp = self.layernorm1b(inp)
        attn_output, _ = self.mha(lat, inp, inp, xdiff_cross, mask=mask)  # (..., latent_size, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        
        latents = latents + attn_output

        lat = self.layernorm2(latents)
        ffn_output = self.ffn[1](self.ffn[0](lat))  # (..., input_size, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        return latents + ffn_output  # (..., latent_size, d_model)
    
    
class XdiffPerciever(tf.keras.layers.Layer):
    def __init__(self, num_outputs, num_layers, num_repeats, latents_per_x, d_cross, d_model, num_heads, dff_input, dff, dff_final, share_weights=False, 
                 activation='gelu', kernel_scale=None, dropout_rate=0.1, scale=1.0):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.num_repeats = num_repeats
        self.d_cross = d_cross
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
        
        if kernel_scale is None:
            kernel_scale = 1/np.sqrt(num_layers*num_repeats+num_layers+num_repeats)
        
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=kernel_scale, mode='fan_avg', distribution='uniform') # scaled Glorot uniform

        self.x_token = self.add_weight(name='x_token', shape=(latents_per_x, d_model), dtype=tf.float32, trainable=True, 
                                       initializer=tf.keras.initializers.RandomNormal(stddev=kernel_scale)) # (d_model)
        
        num_x_features = 26
        self.enc_layers = [[XdiffEncoderLayer(d_model, num_heads, num_x_features, dff, activation=activation, 
                                              dropout_rate=dropout_rate, kernel_initializer=self.kernel_initializer) for _ in range(num_layers)] for _ in range(num_repeats+1)]
        self.cross_enc_layers = [XdiffCrossEncoderLayer(d_cross, d_model, num_heads, num_x_features, dff, dff_input, activation=activation, 
                                                        dropout_rate=dropout_rate, kernel_initializer=self.kernel_initializer) for _ in range(num_repeats)]

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='final_layernorm')
        
        self.pre_final_layers = [tf.keras.layers.Dense(dff, activation=activation, name=f'output_ffn_{d}') for d, dff in enumerate(dff_final)]
        self.final_layer = tf.keras.layers.Dense(num_outputs, name=f'output_dense_layer')
    
    def get_config(self):
        return {
            "num_outputs": self.num_outputs,
            "num_layers": self.num_layers,
            "num_repeats": self.num_repeats,
            "d_model": self.d_model,
            "d_cross": self.d_cross,
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
        xdiff = get_xdiff(x_outputs, x_outputs)/self.scale # (..., latent_size, latent_size, x_features)
        xdiff_cross = get_xdiff(x_outputs, x_inputs)/self.scale # (..., latent_size, input_size, x_features)
        
        latents = x_token
        
        for r in range(self.num_repeats):
            for i in range(self.num_layers):
                latents = self.enc_layers[r][i](latents, xdiff, training=training, mask=None)
            latents = self.cross_enc_layers[r](latents, inputs, xdiff_cross, training=training, mask=mask)

        for i in range(self.num_layers):
            latents = self.enc_layers[self.num_repeats][i](latents, xdiff, training=training, mask=None)
            
        latents = self.layernorm(latents)
        
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
            
        latents = x_token
        
        layers = []
        for r in range(self.num_repeats):
            for i in range(self.num_layers):
                layers.append(self.enc_layers[r][i])
            layers.append(self.cross_enc_layers[r])

        for i in range(self.num_layers):
            layers.append(self.enc_layers[self.num_repeats][i])
            
        layers.append(self.layernorm)
        
        for layer in self.pre_final_layers:
            layers.append(layer)
            
        layers.append(self.final_layer)
        
        return latents, xdiff, inputs, xdiff_cross, layers
