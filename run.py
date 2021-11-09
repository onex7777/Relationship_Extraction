#! -*- coding:utf-8 -*-
# pip install bert4keras==0.7.8
import os, argparse
os.environ["TF_KERAS"] = '1'
import tensorflow as tf
from data_loader import data_generator, load_data
from Bert_pre_train import E2EModels
from utils import get_tokenizer, metric
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--train', default=False, type=bool, help='to train the HBT save_model, python run.py --train=True')
parser.add_argument('--dataset_dir', default='DuIE_2_0/triples',
                    type=str, help='')
args = parser.parse_args()

if __name__ == '__main__':
    # pre-trained bert save_model config
    bert_model = 'weight'
    bert_config_path = bert_model + '/config.json'
    bert_vocab_path = bert_model + '/vocab.txt'
    bert_checkpoint_path = bert_model + '/bert_model.ckpt'

    dataset_dir = args.dataset_dir
    train_path = dataset_dir + '/train_triples.json'
    dev_path = dataset_dir + '/dev_triples.json'
    test_path = dataset_dir + '/test_triples.json'
    rel_dict_path = dataset_dir + '/rel2id.json'
    save_weights_path = './checkpoint/'
    log_dir = "./log_dir/"
    BATCH_SIZE = 32
    tokenizer = get_tokenizer(bert_vocab_path)
    train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_path, dev_path, test_path,
                                                                          rel_dict_path)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-6,
        decay_steps=len(train_data) // BATCH_SIZE // 5,
        decay_rate=0.92, staircase=True)
    subject_model, object_model, hbt_model = E2EModels(lr_schedule, num_rels)

    if args.train:
        MAX_LEN = 256
        train_STEPS = len(train_data) // BATCH_SIZE
        dev_STEPS = len(dev_data) // BATCH_SIZE
        data_manager = data_generator(train_data, tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
        dev_data_manager = data_generator(dev_data, tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
        if not os.path.exists(save_weights_path):
            print('-----make save_model savepath----')
            os.makedirs(save_weights_path)
        else:
            hbt_model.load_weights(os.path.join(save_weights_path, 'best_model.weights'))
            print('-----load save_model weight----')
        logging = TensorBoard(log_dir=log_dir)  # 使用TensorBoard将keras的训练过程显示出来
        # # 每两个周期进行保存，定时储存数据
        # checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}.h5',
        #                              monitor='val_loss', save_weights_only=True, save_best_only=False, period=1)

        # reduce_lr = ReduceLROnPlateau(monitor='val_loss',  # monitor监测的量
        #                               factor=0.5,  # 每次减小学习中的因子，学习率将以Lr=lr*factor的形式减小
        #                               patience=2,  # 每3个epoch过程模型性能不提升，便触发Lr
        #                               verbose=1)
        # # 当6个epoch没有降低loss，则停止
        early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=1)


        hbt_model.fit(data_manager.__iter__(),
                      steps_per_epoch=train_STEPS,
                      epochs=30,
                      initial_epoch=0,
                      validation_freq=2,
                      validation_data=dev_data_manager.__iter__(),
                      validation_steps=dev_STEPS,
                    callbacks=[logging, early_stopping])
        hbt_model.save_weights(os.path.join(save_weights_path, 'best_model.weights'))
        precision, recall, f1 = metric(subject_model, object_model, dev_data, id2rel, tokenizer, name="dev_data")
        print('f1: %.4f, precision: %.4f, recall: %.4f\n' % (f1, precision, recall))
    else:
        hbt_model.load_weights(os.path.join(save_weights_path, 'best_model.weights'))
        test_result_path = 'test_result.json'
        isExactMatch = True if dataset_dir == 'Wiki-KBP' else False

        precision, recall, f1_score = metric(subject_model, object_model, test_data, id2rel, tokenizer, isExactMatch,
                                             test_result_path,name="test_data")
        print(r'precision:{0}\trecall:{1}\tf1_score:{2}'.format(round(precision, 4), round(recall, 4),
                                                                round(f1_score, 4)))
