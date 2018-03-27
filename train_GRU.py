# -*- coding: UTF-8 -*-
import tensorflow as tf
import numpy as np
import time
import datetime
import os
from RE_BGRU_2ATT import network
from tensorflow.contrib.tensorboard.plugins import projector
#RETrainData 里面为最原始的训练样本：相关症状，致病原因，不明确，并发症 688个
#newtrain里面的为：症状疾病，疾病症状，疾病病因，病因疾病，不明确，并发症的样本集 688个， relation2id
#RETrainDataNew 里面为：症状疾病，疾病症状，疾病病因，病因疾病，间接关系，间接关系因，歧义，不明确，并发症的样本集，共824,relationRETRainDataNew
#RETrainDataFinal 里面为：症状疾病，疾病症状，疾病病因，病因疾病，间接关系，歧义，不明确，的样本集，共988, relationRETRainDataFinal
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')


def main(_):
    # the path to save models
    save_path = './model/26/'

    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')

    print('reading training data')
    train_y = np.load('./data/Healthtrain_y.npy')
    train_word = np.load('./data/Healthtrain_word.npy')
    train_pos1 = np.load('./data/Healthtrain_pos1.npy')
    train_pos2 = np.load('./data/Healthtrain_pos2.npy')

    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    #看一下是不是13
    settings.num_classes = len(train_y[0])

    big_num = settings.big_num

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                #如果以后要改网络在这里
                m = network.GRU(is_training=True, word_embeddings=wordembedding, settings=settings)
            #看一下gloabal_step值
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.0005)

            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
           
            merged_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, big_num):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                for i in range(len(word_batch)):
                    #total_shape[i] 记录前i个sentence的长度
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch

                temp, step, loss, accuracy, summary, l2_loss, final_loss = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)

                if step % 200 == 0:
                    tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                    print(tempstr)

            for one_epoch in range(settings.num_epochs):

                temp_order = list(range(len(train_word)))
                np.random.shuffle(temp_order)
                for i in range(int(len(temp_order) / float(settings.big_num))):

                    temp_word = []
                    temp_pos1 = []
                    temp_pos2 = []
                    temp_y = []

                    temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                    for k in temp_input:
                        temp_word.append(train_word[k])
                        temp_pos1.append(train_pos1[k])
                        temp_pos2.append(train_pos2[k])
                        temp_y.append(train_y[k])
                    num = 0
                    for single_word in temp_word:
                        num += len(single_word)

                    #为什么是1500
                    if num > 1500:
                        print('out of range')
                        continue

                    temp_word = np.array(temp_word)
                    temp_pos1 = np.array(temp_pos1)
                    temp_pos2 = np.array(temp_pos2)
                    temp_y = np.array(temp_y)

                    train_step(temp_word, temp_pos1, temp_pos2, temp_y, settings.big_num)

                    current_step = tf.train.global_step(sess, global_step)
                    #if current_step > 8000 and current_step % 100 == 0:
                    if current_step % 1000 == 0 and current_step > 1999:
                        print('saving model')
                        path = saver.save(sess, save_path + 'ATT_GRU_model_Health_', global_step=current_step)
                        tempstr = 'have saved model to ' + path
                        print(tempstr)


if __name__ == "__main__":
    tf.app.run()
