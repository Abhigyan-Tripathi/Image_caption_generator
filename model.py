import tensorflow as tf
import numpy as np

# ... [Paste all your attention layers here: self_attention, MultiHeadAttention, 
# masked_self_attention, masked_multi_head_attention, cross_attention, 
# multi_head_cross_attention] ...

class self_attention(tf.keras.Model):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.wq = tf.keras.layers.Dense(self.d_k)
        self.wk = tf.keras.layers.Dense(self.d_k)
        self.wv = tf.keras.layers.Dense(self.d_k) # 128 in this case
    def call(self, src_embeddings):
        Q = self.wq(src_embeddings)
        K = self.wk(src_embeddings)
        V = self.wv(src_embeddings)
        q_k_dot_product = tf.matmul(Q, K, transpose_b=True)
        scaled_q_k_dot_product = q_k_dot_product / np.sqrt(self.d_k)
        attention_weights = tf.nn.softmax(scaled_q_k_dot_product, axis=-1)
        contextualized_vector = tf.matmul(attention_weights, V)
        return contextualized_vector
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.heads = [self_attention(d_model, num_heads) for _ in range(num_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
    def call(self, src_embeddings):
        head_outputs = [head(src_embeddings) for head in self.heads]   # each: [batch, seq_len, d_k]
        concat = tf.concat(head_outputs, axis=-1)                       # [batch, seq_len, d_model]
        return self.wo(concat)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config

class masked_self_attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.wq = tf.keras.layers.Dense(self.d_k)
        self.wk = tf.keras.layers.Dense(self.d_k)
        self.wv = tf.keras.layers.Dense(self.d_k) # 128 in this case
    def causal_mask(self,seq_len):
        # Upper triangle (excluding diagonal) = 1 -> gets masked; diagonal and below = 0 -> stays visible
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)          # [seq_len, seq_len]
        return tf.cast(np.where(mask == 1, -np.inf, 0.0), tf.float32)
    def call(self, src_embeddings):
        Q = self.wq(src_embeddings)
        K = self.wk(src_embeddings)
        V = self.wv(src_embeddings)
        q_k_dot_product = tf.matmul(Q, K, transpose_b=True)
        scaled_q_k_dot_product = q_k_dot_product / np.sqrt(self.d_k)
        masked_scaled_q_k_dot_product = scaled_q_k_dot_product + self.causal_mask(scaled_q_k_dot_product.shape[-1])
        attention_weights = tf.nn.softmax(masked_scaled_q_k_dot_product, axis=-1)
        contextualized_vector = tf.matmul(attention_weights, V)
        return contextualized_vector
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config

class masked_multi_head_attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.heads = [masked_self_attention(d_model, num_heads) for _ in range(num_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
    def call(self, src_embeddings):
        head_outputs = [head(src_embeddings) for head in self.heads]
        concat = tf.concat(head_outputs, axis=-1)                       # [batch, seq_len, d_model]
        return self.wo(concat)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config

class cross_attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.wq = tf.keras.layers.Dense(self.d_k)
        self.wk = tf.keras.layers.Dense(self.d_k)
        self.wv = tf.keras.layers.Dense(self.d_k)
    def call(self, src_embeddings, tgt_embeddings):
        Q = self.wq(tgt_embeddings)
        K = self.wk(src_embeddings)
        V = self.wv(src_embeddings)
        q_k_dot_product = tf.matmul(Q, K, transpose_b=True)
        scaled_q_k_dot_product = q_k_dot_product / np.sqrt(self.d_k)
        attention_weights = tf.nn.softmax(scaled_q_k_dot_product, axis=-1)
        contextualized_vector = tf.matmul(attention_weights, V)
        return contextualized_vector
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config

class multi_head_cross_attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.heads = [cross_attention(d_model, num_heads) for _ in range(num_heads)]
        self.wo = tf.keras.layers.Dense(d_model)
    def call(self, src_embeddings, tgt_embeddings):
        head_outputs = [head(src_embeddings, tgt_embeddings) for head in self.heads]
        concat = tf.concat(head_outputs, axis=-1)                       # [batch, seq_len, d_model]
        return self.wo(concat)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config


class ffn(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.fc2 = tf.keras.layers.Dense(d_model)
    def call(self, x):
        return self.fc2(self.fc1(x))
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "d_ff": self.d_ff})
        return config

class ImageEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model=512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        self.resnet.trainable = False
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(d_model, activation='linear')
        ])
    def call(self, images):
        features = self.resnet(images)
        batch_size = tf.shape(features)[0]
        features = tf.reshape(features, (batch_size, 49, 2048))
        return self.ffn(features)
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config

class Decoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        # Initialize your attention and ffn layers here...
        self.masked_self_attention = masked_multi_head_attention(d_model, num_heads)
        self.cross_attention = multi_head_cross_attention(d_model, num_heads)
        self.ffn_layer = ffn(d_model, d_ff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, target_embeddings, encoder_output):
        masked_contextual_input = self.masked_self_attention(target_embeddings)
        output = self.layernorm1(target_embeddings + masked_contextual_input)
        cross_attention_output = self.cross_attention(encoder_output, output)
        output = self.layernorm2(output + cross_attention_output)
        ffn_output = self.ffn_layer(output)
        return self.layernorm3(output + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads, "d_ff": self.d_ff})
        return config

class ImageCaptioner(tf.keras.Model):
    def __init__(self, d_model, num_heads, d_ff, target_vocab_size, max_len=30, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.target_vocab_size = target_vocab_size
        self.max_len = max_len
        
        self.image_encoder = ImageEncoder(d_model)
        self.tgt_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, mask_zero=True)
        
        # Positional Encoding
        positions = np.arange(max_len)[:, np.newaxis]
        i = np.arange(d_model // 2)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * i) / d_model)
        angles = positions * angle_rates
        pos_encoding = np.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angles)
        pos_encoding[:, 1::2] = np.cos(angles)
        self.tgt_pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)
        
        self.decoder1 = Decoder(d_model, num_heads, d_ff)
        self.decoder2 = Decoder(d_model, num_heads, d_ff)
        self.decoder3 = Decoder(d_model, num_heads, d_ff)
        self.linear = tf.keras.layers.Dense(target_vocab_size)

    def call(self, images, tgt):
        encoder_output = self.image_encoder(images)
        tgt_seq_len = tf.shape(tgt)[1]
        tgt_embeddings = self.tgt_embedding(tgt)
        tgt_embeddings = tgt_embeddings + self.tgt_pos_encoding[tf.newaxis, :tgt_seq_len, :]
        
        dec_out1 = self.decoder1(tgt_embeddings, encoder_output)
        dec_out2 = self.decoder2(dec_out1, encoder_output)
        dec_out3 = self.decoder3(dec_out2, encoder_output)
        return self.linear(dec_out3)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model, "num_heads": self.num_heads, 
            "d_ff": self.d_ff, "target_vocab_size": self.target_vocab_size,
            "max_len": self.max_len
        })
        return config