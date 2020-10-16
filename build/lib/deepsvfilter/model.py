# coding=utf-8
from __future__ import print_function

from operator import itemgetter
import os
import time
import random
import sys
from sklearn.metrics import roc_curve, auc
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageDraw
from utils.evaluation import *

slim = tf.contrib.slim

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from cnns.mobilenet_v1.mobilenet_v1 import *
from cnns.mobilenet_v2.mobilenet_v2 import *
from cnns.nasnet_mobile.nasnet import *
from cnns.pnasnet_5_mobile.pnasnet import *
from cnns.inception_resnet_v2.inception_resnet_v2 import *


def data_augmentation(image, mode, patch_size):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        image1 = np.zeros(image.shape, dtype="float32")
        image = np.fliplr(image)
        image1[:patch_size // 2 - 1, :, :] = image[patch_size - patch_size // 2 + 1:, :, :]
        image1[patch_size // 2 - 1:patch_size - patch_size // 2 + 1, :, :] = \
            image[patch_size // 2 - 1:patch_size - patch_size // 2 + 1, :, :]
        image1[patch_size - patch_size // 2 + 1:, :, :] = image[:patch_size // 2 - 1, :, :]
        return image1


class model(object):
    def __init__(self, sess, is_training, model_name):
        self.sess = sess

        # build the model
        self._num_classes = 2
        self._model_name = model_name
        if self._model_name == "Inception_ResNet_v2":
            self._patch_size = 299
        else:
            self._patch_size = 224
        self._is_training = is_training
        self.input_im = tf.placeholder(tf.float32, [None, self._patch_size, self._patch_size, 3], name='input_im')
        self.input_label = tf.placeholder(tf.float32, [None, self._num_classes], name='input_y')
        self._keep_prob = tf.placeholder(tf.float32)

        # Predict : Predict prediction tensors from inputs tensor.
        if self._model_name == "Inception_ResNet_v2":
            net, net2, self.logits = self.Inception_ResNet(self.input_im, keep_prob=self._keep_prob)
        elif self._model_name == "MobileNet_v1":
            net, net2, self.logits = self.MobileNet1(self.input_im, keep_prob=self._keep_prob)
        elif self._model_name == "MobileNet_v2_1.0":
            net, net2, self.logits = self.MobileNet2(self.input_im, keep_prob=self._keep_prob)
        elif self._model_name == "MobileNet_v2_1.4":
            net, net2, self.logits = self.MobileNet2(self.input_im, keep_prob=self._keep_prob, multiplier=1.4)
        elif self._model_name == "NASNet_A_Mobile":
            net, net2, self.logits = self.NASNet(self.input_im, keep_prob=self._keep_prob)
        else:
            net, net2, self.logits = self.PNASNet(self.input_im, keep_prob=self._keep_prob)

        print("Shape of ", model_name, " Out net: ", net.shape, " Out net2: ", net2.shape)
        self.input_classes = tf.argmax(self.input_label, axis=1, name="input_classes")  # 正例是0，负例是1

        # restore variables
        checkpoint_exclude_scopes = 'Logits'
        exclusions = None
        if checkpoint_exclude_scopes:
            exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
        variables_to_restore = []
        for var in slim.get_model_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                variables_to_restore.append(var)

        # Loss : Compute scalar loss tensors with respect to provided groundtruth.
        self.recon_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                                     labels=self.input_label, name='softmax_loss')
        self.loss_SV = tf.reduce_mean(self.recon_loss, name='recon_loss')  # + l2_loss

        # Accuracy : Calculate accuracy.
        self.logits = tf.nn.softmax(self.logits)
        self.prediction = tf.argmax(self.logits, axis=1, name="predictions")
        # shape of prediction = (16,)
        correct_predictions = tf.equal(tf.cast(self.prediction, tf.int32),
                                       tf.cast(tf.argmax(self.input_label, 1), tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        tf.summary.scalar('loss', self.loss_SV)
        tf.summary.scalar('accuracy', self.accuracy)

        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        # self.var_SV = [var for var in tf.trainable_variables() if 'SVNet' in var.name]
        self.var_SV = tf.global_variables()
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.train_op_SV = optimizer.minimize(self.loss_SV, var_list=self.var_SV)
        self.train_op_SV = slim.learning.create_train_op(self.loss_SV, optimizer)

        self.merged = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())

        self.saver_restore = tf.train.Saver(var_list=variables_to_restore)
        self.saver = tf.train.Saver(var_list=self.var_SV)

        print("[*] Initialize model successfully...")

    def MobileNet1(self, input_im, keep_prob):
        """ Predict prediction tensors from inputs tensor.
        Outputs of this function can be passed to loss or postprocess functions.
        Args:
            input_im: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        print("=====================================[", self._model_name, "] init==================================")
        with slim.arg_scope(mobilenet_v1_arg_scope()):
            net, endpoints = mobilenet_v1(input_im, num_classes=None, is_training=self._is_training)
        # MobilenetV1/Conv2d_13_pointwise/BatchNorm/beta:0 (1024,)
        with tf.variable_scope('Logits'):
            net1 = tf.squeeze(net, axis=[1, 2])  #
            net2 = slim.dropout(net1, keep_prob=keep_prob, scope='scope')
            logits = slim.fully_connected(net2, num_outputs=self._num_classes, activation_fn=None, scope='_Predict')
            # shape(logits) = (16,2)
        return net, net2, logits

    def MobileNet2(self, input_im, keep_prob, multiplier=1.0):
        print("==========================[", self._model_name, "] init=======================")
        with slim.arg_scope(training_scope(is_training=self._is_training)):
            net, endpoints = mobilenet(input_im, num_classes=None, depth_multiplier=multiplier)
            # 1.0: MobilenetV2/Conv_1/BatchNorm/beta:0 (1280,)
            # 1.4: MobilenetV2/Conv_1/BatchNorm/beta:0 (1792,)
        with tf.variable_scope('Logits'):
            net1 = tf.squeeze(net, axis=[1, 2])  #
            net2 = slim.dropout(net1, keep_prob=keep_prob, scope='scope')
            logits = slim.fully_connected(net2, num_outputs=self._num_classes, activation_fn=None, scope='_Predict')
            # shape(logits) = (16,2)
        return net, net2, logits

    def NASNet(self, input_im, keep_prob):
        print("=====================================[", self._model_name, "] init==================================")
        with slim.arg_scope(nasnet_mobile_arg_scope()):
            net, endpoints = build_nasnet_mobile(input_im, num_classes=None, is_training=self._is_training)
            # cell_11/comb_iter_4/left/bn_sep_3x3_2/beta:0 (176,)
        with tf.variable_scope('Logits'):
            net2 = slim.dropout(net, keep_prob=keep_prob, scope='scope')
            logits = slim.fully_connected(net2, num_outputs=self._num_classes, activation_fn=None, scope='_Predict')
            # shape(logits) = (16,2)
        return net, net2, logits

    def PNASNet(self, input_im, keep_prob):
        print("=====================================[", self._model_name, "] init==================================")
        with slim.arg_scope(pnasnet_mobile_arg_scope()):  # pnasnet_large_arg_scope()
            net, endpoints = build_pnasnet_mobile(input_im, num_classes=None, is_training=self._is_training)
            # cell_8/comb_iter_4/left/bn_sep_3x3_2/beta:0 (216,)
        with tf.variable_scope('Logits'):
            net2 = slim.dropout(net, keep_prob=keep_prob, scope='scope')
            logits = slim.fully_connected(net2, num_outputs=self._num_classes, activation_fn=None, scope='_Predict')
            # shape(logits) = (16,2)
        return net, net2, logits

    def Inception_ResNet(self, input_im, keep_prob):
        print("=====================================[", self._model_name, "] init==================================")
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            net, endpoints = inception_resnet_v2(input_im, num_classes=None, is_training=self._is_training)
            # InceptionResnetV2/Conv2d_7b_1x1/BatchNorm/beta:0 (1536,)
        with tf.variable_scope('Logits'):
            net1 = tf.squeeze(net, axis=[1, 2])  #
            net2 = slim.dropout(net1, keep_prob=keep_prob, is_training=self._is_training, scope='scope')
            logits = slim.fully_connected(net2, num_outputs=self._num_classes, activation_fn=None, scope='_Predict')
            # shape(logits) = (16,2)
        return net, net2, logits

    def evaluate(self, epoch_num, eval_im, eval_label, batch_size, patch_size, iter_num, epoch):
        print("[*] Evaluating for epoch %d..." % (epoch_num))

        batch_input_im = np.zeros((len(eval_im), patch_size, patch_size, 3), dtype="float32")
        batch_input_y = np.zeros((len(eval_im), 2), dtype="float32")
        image_id = 0
        for patch_id in range(len(eval_im)):  # 16
            batch_input_im[patch_id, :, :, :] = eval_im[image_id]
            batch_input_y[patch_id, :] = eval_label[image_id]

            image_id = (image_id + 1) % len(eval_im)
            if image_id == 0:
                tmp = list(zip(eval_im, eval_label))
                random.shuffle(tmp)
                eval_im, eval_label = zip(*tmp)

        eval_predict_label, eval_predict_loss, eval_accuracy, input_classes, logits = self.sess.run(
            [self.prediction, self.loss_SV, self.accuracy, self.input_classes, self.logits],
            feed_dict={self.input_im: batch_input_im,
                       self.input_label: batch_input_y,
                       self._keep_prob: 1})

        true_label = [abs(num - 1.) for num in input_classes.tolist()]
        predict_score = [logit[0] for logit in logits.tolist()]

        input_classes = [str(a) for a in input_classes.tolist()]
        predict_label = [str(a) for a in eval_predict_label.tolist()]

        count_tp, count_tn, count_fn, count_fp, \
        tpr, fpr, tnr, acc, precision, recall, f1, roc_auc = evaluation1(true_label, predict_score,
                                                                         input_classes, predict_label)
        print(
            "Eval_predict_loss=%.6f, auc=%.4f, TPR=%.4f, FPR=%.4f, Eval_accuracy=%.4f, TNR=%.4f, Precision=%.4f, Recall=%.4f, F1-score=%.4f" % (
                eval_predict_loss, roc_auc, tpr, fpr, eval_accuracy, tnr, precision, recall, f1))
        return eval_accuracy

    def train(self, train_im, train_label, eval_im, eval_label, batch_size, patch_size, epoch, eval_every_epoch, lr,
              ckpt_dir, summary_dir):

        numBatch = len(train_im) // int(batch_size)  # 20000//16=1250

        # load pretrained model
        train_op = self.train_op_SV
        train_loss = self.loss_SV
        saver = self.saver
        saver_restore = self.saver_restore

        # load_model_status, global_step = self.load(saver, ckpt_dir)
        load_model_status, global_step = self.load(saver_restore, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!", global_step, numBatch, start_epoch, start_step)
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("Print the trainable parameters:")
        for eval_ in tf.trainable_variables():
            w_val = self.sess.run(eval_.name)
            print(eval_.name, w_val.shape)

        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        print("[*] Start training with start epoch %d start iter %d : " % (start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        max_eval_accuracy=0
        for epoch in range(start_epoch, epoch):  # epoch=20
            for batch_id in range(start_step, numBatch):  # num of batch = 10000//16=625
                # generate data for a batch
                batch_input_im = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                # batch_input_y = np.zeros((batch_size, 2), dtype="float32")
                batch_input_y = np.zeros((batch_size, 2), dtype="float32")
                for patch_id in range(batch_size):
                    rand_mode = random.randint(0, 1)  #
                    # batch_input_im[patch_id, :, :, :] = data_augmentation(train_im[image_id], rand_mode, patch_size)
                    batch_input_im[patch_id, :, :, :] = train_im[image_id]
                    batch_input_y[patch_id, :] = train_label[image_id]
                    image_id = (image_id + 1) % len(train_im)
                    if image_id == 0:
                        tmp = list(zip(train_im, train_label))
                        random.shuffle(tmp)
                        train_im, train_label = zip(*tmp)

                # train
                _, loss, acc, input_classes, output_predictions, summary = self.sess.run(
                    [train_op, train_loss, self.accuracy, self.input_classes, self.prediction, self.merged],
                    feed_dict={self.input_im: batch_input_im,  # [16,224,224,6]
                               self.input_label: batch_input_y,  # [16,2]
                               self._keep_prob: 0.5,
                               self.lr: lr[epoch]})

                summary_writer.add_summary(summary, batch_id)

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, accuracy: %.4f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss, acc), ', input: ',
                      "".join([str(a) for a in input_classes.tolist()]),
                      ' , predict: ', "".join([str(a) for a in output_predictions.tolist()]))
                iter_num += 1
            # evalutate the model and save a checkpoint file for it
            if (int(epoch) + 1) % int(eval_every_epoch) == 0:
                
                current_eval_accuracy=self.evaluate(epoch + 1, eval_im, eval_label, batch_size, patch_size, iter_num, epoch)  # Eval set test
                # self.evaluate(epoch + 1, test_im, test_label, batch_size, patch_size, iter_num, epoch)  # NA12878 test
                if current_eval_accuracy > max_eval_accuracy:
                    self.save(saver, iter_num, ckpt_dir, "DeepSVFilter_" + self._model_name)
                    max_eval_accuracy=current_eval_accuracy

        summary_writer.close()
        print("[*] Finish training.")

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name, str(iter_num))
        saver.save(self.sess,
                   os.path.join(ckpt_dir, model_name),
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                print("ckpt_dir=", ckpt_dir)
                print("full_path=", full_path)
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def predict(self, test_im, test_names, patch_size, ckpt_dir, res_dir):
        tf.global_variables_initializer().run()
        assert len(test_im) == len(test_names)
        print("[*] Reading checkpoint...")
        load_model_status_SV, _global_step = self.load(self.saver, ckpt_dir)
        if load_model_status_SV:
            print("[*] Load weights successfully...(iter_num=", _global_step, ")")

        print("[*] Testing...")

        batch_input_im = np.zeros((len(test_im), patch_size, patch_size, 3), dtype="float32")
        for patch_id in range(len(test_im)):  # 16
            batch_input_im[patch_id, :, :, :] = test_im[patch_id]

        test_predict_label, logits = self.sess.run(
            [self.prediction, self.logits], feed_dict={self.input_im: batch_input_im, self._keep_prob: 1})

        predict_score = [logit[0] for logit in logits.tolist()]

        predict_name = ["Deletion" if int(a) == 0 else "Non_Deletion" for a in test_predict_label.tolist()]
        
        chrome_list = [a.split('/')[-1].split('_')[1] for a in test_names]
        start_list = [a.split('/')[-1].split('_')[2] for a in test_names]
        end_list = [a.split('/')[-1].split('_')[3].split('.')[0] for a in test_names]

        res_f = open(os.path.join(res_dir, "results.txt"), 'w')
       
        result_list=list(zip(chrome_list, start_list, end_list, predict_name, predict_score))

        table_sorted = sorted(result_list, key=itemgetter(0, 1))

        for row in table_sorted:
            if(row[4]>=0.5):
                line = "%s\t%s\t%s\t%s\t%.4f" % (row[0], row[1], row[2], row[3], row[4])
                res_f.write(line + '\n')
        
