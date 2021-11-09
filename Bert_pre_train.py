import os
os.environ["TF_KERAS"] = '1'
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from bert4keras.backend import set_gelu  # pip install bert4keras==0.7.8
from bert4keras.models import build_transformer_model
from utils import seq_gather
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

config_path = 'D:\\Python\\Tensorflow\\natural_language_processing\\Relationship_Extraction\\weight\\bert_config.json'
checkpoint_path = 'D:\\Python\\Tensorflow\\natural_language_processing\\Relationship_Extraction\\weight\\bert_model.ckpt'
set_gelu('tanh')


def E2EModels(lr, num_rels):
    bert_model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model='bert',
        return_keras_model=True)
    gold_sub_heads_in = tf.keras.layers.Input(shape=(None,))  # # [batch,len]
    gold_sub_tails_in = tf.keras.layers.Input(shape=(None,))
    sub_head_in = tf.keras.layers.Input(shape=(1,))  # [batch,1]
    sub_tail_in = tf.keras.layers.Input(shape=(1,))
    gold_obj_heads_in = tf.keras.layers.Input(shape=(None, num_rels))  # [batch,len,num_rels]
    gold_obj_tails_in = tf.keras.layers.Input(shape=(None, num_rels))

    gold_sub_heads, gold_sub_tails, sub_head, sub_tail, gold_obj_heads, gold_obj_tails =\
    gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in, gold_obj_heads_in, gold_obj_tails_in

    tokens = bert_model.input[0]
    # K.greater逐元素比较，返回bool
    mask = tf.keras.layers.Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(tokens)
    # output_layer = 'Transformer-2-FeedForward-Norm'
    # tokens_feature = bert_model.get_layer(output_layer).output
    tokens_feature = bert_model.output
    pred_sub_heads = tf.keras.layers.Dense(1, activation='sigmoid')(tokens_feature)
    pred_sub_tails = tf.keras.layers.Dense(1, activation='sigmoid')(tokens_feature)

    subject_model = Model(bert_model.input, [pred_sub_heads, pred_sub_tails])

    sub_head_feature = tf.keras.layers.Lambda(seq_gather)([tokens_feature, sub_head])
    sub_tail_feature = tf.keras.layers.Lambda(seq_gather)([tokens_feature, sub_tail])
    sub_feature = tf.keras.layers.Average()([sub_head_feature, sub_tail_feature])

    tokens_feature = tf.keras.layers.Add()([tokens_feature, sub_feature])
    pred_obj_heads = tf.keras.layers.Dense(num_rels, activation='sigmoid')(tokens_feature)
    pred_obj_tails = tf.keras.layers.Dense(num_rels, activation='sigmoid')(tokens_feature)

    object_model = Model(bert_model.input + [sub_head_in, sub_tail_in], [pred_obj_heads, pred_obj_tails])

    hbt_model = Model(
        bert_model.input + [gold_sub_heads_in, gold_sub_tails_in, sub_head_in, sub_tail_in, gold_obj_heads_in,
                            gold_obj_tails_in],
        [pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails])

    gold_sub_heads = K.expand_dims(gold_sub_heads, 2)
    gold_sub_tails = K.expand_dims(gold_sub_tails, 2)
    # gold_sub_heads类似于one——hot编码，将其他非为所需标签作为负样本
    sub_heads_loss = K.binary_crossentropy(gold_sub_heads, pred_sub_heads)  # 使得采用2分类大概率预测的值为所需的标签概率
    sub_heads_loss = K.sum(sub_heads_loss * mask) / K.sum(mask)
    sub_tails_loss = K.binary_crossentropy(gold_sub_tails, pred_sub_tails)
    sub_tails_loss = K.sum(sub_tails_loss * mask) / K.sum(mask)

    obj_heads_loss = K.sum(K.binary_crossentropy(gold_obj_heads, pred_obj_heads), 2, keepdims=True)
    obj_heads_loss = K.sum(obj_heads_loss * mask) / K.sum(mask)
    obj_tails_loss = K.sum(K.binary_crossentropy(gold_obj_tails, pred_obj_tails), 2, keepdims=True)
    obj_tails_loss = K.sum(obj_tails_loss * mask) / K.sum(mask)
    # alpha = tf.Variable(initial_value=0.7, trainable=True,name="alpha")
    # beta = tf.Variable(initial_value=0.3, trainable=True,name="beta")
    # loss = (alpha/(alpha+beta))*(sub_heads_loss + sub_tails_loss) + (beta/(alpha+beta))*(obj_heads_loss + obj_tails_loss)
    loss = (sub_heads_loss + sub_tails_loss) +( obj_heads_loss + obj_tails_loss)
    hbt_model.add_loss(loss)
    hbt_model.compile(optimizer=Adam(lr))
    hbt_model.summary()
    print("the num of layers:", len(hbt_model.layers))
    # plot_model(hbt_model, to_file='./save_model.png', show_shapes=True)
    return subject_model, object_model, hbt_model


if __name__ == "__main__":
    E2EModels(0.01, 55)
