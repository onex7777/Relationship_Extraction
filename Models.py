import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import keras.backend as K
from utils import seq_gather, metric
import tensorflow_addons as tfa
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.utils import plot_model


class MultiHead(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.wq = keras.layers.Dense(self.n_head * self.head_dim)
        self.wk = keras.layers.Dense(self.n_head * self.head_dim)
        self.wv = keras.layers.Dense(self.n_head * self.head_dim)  # [n, step, h*h_dim]

        self.o_dense = keras.layers.Dense(self.model_dim)
        self.o_drop = keras.layers.Dropout(rate=drop_rate)
        # self.attention = None

    def call(self, q, k, v, mask, training):
        _q = self.wq(q)  # [n, q_step, h*h_dim]
        _k, _v = self.wk(k), self.wv(v)  # [n, step, h*h_dim]
        _q = self.split_heads(_q)  # [n, h, q_step, h_dim]
        _k, _v = self.split_heads(_k), self.split_heads(_v)  # [n, h, step, h_dim]
        context = self.scaled_dot_product_attention(_q, _k, _v, mask)  # [n, q_step, h*dv]
        o = self.o_dense(context)  # [n, step, dim]
        o = self.o_drop(o, training=training)
        return o

    def split_heads(self, x):
        # 搭建模型查看参数时，原shape值不为None的，需要用x.shape来取值,为None的需要用tf.shape来取值
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], self.n_head, self.head_dim))  # [n, step, h, h_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])  # [n, h, step, h_dim]

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = tf.cast(k.shape[-1], dtype=tf.float32)
        score = tf.matmul(q, k, transpose_b=True) / (tf.math.sqrt(dk) + 1e-8)  # [n, h, q_step, step]
        if mask is not None:
            score += mask * -1e9
        self.attention = tf.nn.softmax(score, axis=-1)  # [n, h, q_step, step]
        context = tf.matmul(self.attention, v)  # [n, h, q_step, step] @ [n, h, step, dv] = [n, h, q_step, dv]
        context = tf.transpose(context, perm=[0, 2, 1, 3])  # [n, q_step, h, dv]
        # 搭建模型查看参数时，原shape值不为None的，需要用x.shape来取值,为None的需要用tf.shape来取值，且在reshape时，不要使用-1
        context = tf.reshape(context, (
        tf.shape(context)[0], tf.shape(context)[1], context.shape[2] * context.shape[3]))  # [n, q_step, h*dv]
        return context


class GELU(layers.Layer):
    def __init__(self):
        super(GELU, self).__init__()

    def call(self, x):
        cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf


# 为了重新定义结果的shape，方便传入下一层layer合并
class PositionWiseFFN(keras.layers.Layer):
    def __init__(self, model_dim):
        super().__init__()
        dff = model_dim * 4
        self.l = keras.layers.Dense(dff)
        self.activation = GELU()
        self.o = keras.layers.Dense(model_dim)

    def call(self, x):
        o = self.l(x)
        o = self.activation(o)
        o = self.o(o)
        return o  # [n, step, dim]


class EncodeLayer(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.ln = [keras.layers.LayerNormalization(axis=-1) for _ in range(2)]  # only norm z-dim
        self.mh = MultiHead(n_head, model_dim, drop_rate)
        self.ffn = PositionWiseFFN(model_dim)
        self.drop = keras.layers.Dropout(drop_rate)

    def call(self, xz, training, mask):
        attn = self.mh.call(xz, xz, xz, mask, training)  # [n, step, dim]
        o1 = self.ln[0](attn + xz)
        ffn = self.drop(self.ffn.call(o1), training)
        o = self.ln[1](ffn + o1)  # [n, step, dim]
        return o


class Encoder(keras.layers.Layer):
    def __init__(self, n_head, model_dim, drop_rate, n_layer):
        super().__init__()
        self.ls = [EncodeLayer(n_head, model_dim, drop_rate) for _ in range(n_layer)]

    def call(self, xz, training, mask):
        for l in self.ls:
            xz = l.call(xz, training, mask)
        return xz  # [n, step, dim]


# 可学习位置编码信息
class PositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, hidden_size, name):
        super(PositionEmbedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=hidden_size,
                                                   embeddings_initializer=tf.initializers.RandomNormal(0., 0.01))

    def call(self, inputs):
        position_ids = tf.range(tf.shape(inputs)[1], dtype=tf.int32)[tf.newaxis, :]
        position_embeddings = self.embedding(position_ids)
        return position_embeddings


class BertEncoder(keras.Model):
    def __init__(self, model_dim, n_layer, n_head, n_vocab, max_len, max_seg=2, drop_rate=0.1, padding_idx=0):
        super(BertEncoder, self).__init__()
        self.padding_idx = padding_idx
        self.n_vocab = n_vocab
        self.max_len = max_len
        self.word_emb = keras.layers.Embedding(
            input_dim=n_vocab, output_dim=model_dim,  # [n_vocab, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01), name="token_embedding"
        )
        self.segment_emb = keras.layers.Embedding(
            input_dim=max_seg, output_dim=model_dim,  # [max_seg, dim]
            embeddings_initializer=tf.initializers.RandomNormal(0., 0.01), name="segment_embedding"
        )
        self.position_emb = PositionEmbedding(max_len, model_dim, name="position_embedding")
        self.encoder = Encoder(n_head, model_dim, drop_rate, n_layer)

    def call(self, seqs, segs, training=False):
        embed = self.input_emb(seqs, segs)  # [n, step, dim]
        pad_mask = self._pad_mask(seqs)
        z = self.encoder(embed, training=training, mask=pad_mask)  # [n, step, dim]
        return z

    def input_emb(self, seqs, segs):
        return self.word_emb(seqs) + self.segment_emb(segs) + self.position_emb(seqs)  # [n, step, dim]

    def _pad_bool(self, seqs):
        return tf.math.equal(seqs, self.padding_idx)

    def _pad_mask(self, seqs):
        mask = tf.cast(self._pad_bool(seqs), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # (n, 1, 1, step)


class Crf_layer(layers.Layer):
    def __init__(self, label_nums):
        super().__init__(self)
        self.label_num = label_nums
        self.dense = layers.Dense(self.label_num, use_bias=False, trainable=True,
                                  kernel_initializer=tf.keras.initializers.GlorotNormal())

    def call(self, inputs, targets, lens, training=None):
        out = self.dense(inputs)  # 调整大小为[batch_size,maxlen,nums_label]
        self.log_likelihood, self.tran_paras = \
            tfa.text.crf_log_likelihood(out, targets, lens)
        self.batch_pred_sequence, self.batch_viterbi_score = tfa.text.crf_decode(out, self.tran_paras, lens)
        self.loss = tf.reduce_mean(-self.log_likelihood)
        return self.loss, self.batch_pred_sequence


def E2EModels(max_len, lr, num_rels):
    bert = BertEncoder(model_dim=768, n_layer=3, n_head=6, n_vocab=21128, max_len=max_len)
    input_token = tf.keras.layers.Input(shape=(None,))
    input_segment = tf.keras.layers.Input(shape=(None,))
    bert_output = bert.call(input_token, input_segment)
    print(bert_output.shape)
    pad_mask = 1. - tf.cast(bert._pad_bool(tf.expand_dims(input_token, 2)), tf.float32)  # 未padding的赋值为一
    print(pad_mask)
    gold_sub_heads_in = tf.keras.layers.Input(shape=(None,))
    gold_sub_tails_in = tf.keras.layers.Input(shape=(None,))
    sub_head_in = tf.keras.layers.Input(shape=(1,))  # [batch,1],batch为每句话随机的一个sub的头坐标
    sub_tail_in = tf.keras.layers.Input(shape=(1,))  # [batch,1],batch为每句话随机的一个sub的尾坐标
    gold_obj_heads_in = tf.keras.layers.Input(shape=(None, num_rels))
    gold_obj_tails_in = tf.keras.layers.Input(shape=(None, num_rels))
    gold_sub_heads, gold_sub_tails, sub_head, sub_tail, gold_obj_heads, gold_obj_tails = \
        gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in, gold_obj_heads_in, gold_obj_tails_in
    pred_sub_heads = keras.layers.Dense(1, activation='sigmoid')(bert_output)
    pred_sub_tails = keras.layers.Dense(1, activation='sigmoid')(bert_output)
    # bert_model.input包含word Embedding和Segment Embedding
    subject_model = Model([input_token, input_segment], [pred_sub_heads, pred_sub_tails])
    sub_head_feature = keras.layers.Lambda(seq_gather)([bert_output, sub_head])  # 得到每个batch对应sub首位置的权值【batch，dim】
    sub_tail_feature = keras.layers.Lambda(seq_gather)([bert_output, sub_tail])  # 得到每个batch对应sub尾位置的权值【batch，dim】
    sub_feature = keras.layers.Average()([sub_head_feature, sub_tail_feature])

    tokens_feature = keras.layers.Add()([bert_output, sub_feature])
    pred_obj_heads = keras.layers.Dense(num_rels, activation='sigmoid')(tokens_feature)
    pred_obj_tails = keras.layers.Dense(num_rels, activation='sigmoid')(tokens_feature)

    object_model = Model([input_token, input_segment] + [sub_head_in, sub_tail_in], [pred_obj_heads, pred_obj_tails])

    hbt_model = Model(
        [input_token, input_segment] + [gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in,
                                        gold_obj_heads_in,
                                        gold_obj_tails_in],
        [pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails])

    gold_sub_heads = K.expand_dims(gold_sub_heads, 2)
    gold_sub_tails = K.expand_dims(gold_sub_tails, 2)

    sub_heads_loss = K.binary_crossentropy(gold_sub_heads, pred_sub_heads)
    sub_heads_loss = K.sum(sub_heads_loss * pad_mask) / K.sum(pad_mask)
    sub_tails_loss = K.binary_crossentropy(gold_sub_tails, pred_sub_tails)
    sub_tails_loss = K.sum(sub_tails_loss * pad_mask) / K.sum(pad_mask)
    x = K.binary_crossentropy(gold_obj_heads, pred_obj_heads)
    obj_heads_loss = K.sum(K.binary_crossentropy(gold_obj_heads, pred_obj_heads), 2, keepdims=True)
    obj_heads_loss = K.sum(obj_heads_loss * pad_mask) / K.sum(pad_mask)
    obj_tails_loss = K.sum(K.binary_crossentropy(gold_obj_tails, pred_obj_tails), 2, keepdims=True)
    obj_tails_loss = K.sum(obj_tails_loss * pad_mask) / K.sum(pad_mask)

    loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)

    hbt_model.add_loss(loss)
    hbt_model.compile(optimizer=keras.optimizers.Adam(lr))
    hbt_model.summary()
    print("the num of layers:", len(hbt_model.layers))
    plot_model(hbt_model, to_file='./model.png', show_shapes=True)
    return subject_model, object_model, hbt_model


if __name__ == "__main__":
    E2EModels(256, 1e-4, 55)

# m = BertEncoder(model_dim=768, max_len=256, n_layer=12, n_head=6, n_vocab=100000)
# x = Input(shape=(None,))  # shape为句子的长度
# y = Input(shape=(None,))
# out = m.call(x, y)
# save_model = keras.Model((x, y), out)
# save_model.summary()
# print("the num of layers:", len(save_model.layers))
