from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.template import loader
import tensorflow as tf
import keras as kr
from django.core.serializers.json import json
# Create your views here.

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 300  # 词向量维度
    seq_length = 30 # 序列长度
    num_classes = 22  # 类别数
    num_filters = 128  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5187  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 1000  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config
        sequence_length = self.config.seq_length
        num_classes = self.config.num_classes
        vocab_size = self.config.vocab_size
        embedding_size = self.config.embedding_dim
        filter_sizes = [1,2,3,4,5,6]
        num_filters = self.config.num_filters
        l2_reg_lambda = 0.0
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.y_pred_cls = tf.argmax(self.logits, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            self.acc = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("predict"):
            self.classIndex = self.y_pred_cls

# words,word_to_id,classList,cat_to_id
def get_words_cat_dir():
    words = []
    with open('cnnModel/data/cnews.vocab.txt', encoding="UTF-8") as f:
        words.extend(f.readlines())
        words = [i.strip() for i in words]

    word_to_id = dict(zip(words, range(len(words))))
    classList = ['phone', 'weather', 'translation', 'playcontrol', 'volume', 'FM',
                 'limitLine', 'alarm', 'schedule', 'music', 'story',
                 'news', 'collect', 'musicinfo', 'healthAI', 'calculator', 'cookbook',
                 'dictionary', 'joke', 'forex', 'stock', 'other']
    cat_to_id = dict(zip(classList, range(len(classList))))
    return words,word_to_id,classList,cat_to_id


def process_one_strLine(strLine,word_to_id, max_length=60):
    data_id = [[word_to_id[x] for x in strLine if x in word_to_id]]
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, truncating='post', padding='post')
    return x_pad


print('Configuring CNN model...')
config = TCNNConfig()
words, word_to_id, classList, cat_to_id = get_words_cat_dir()
config.vocab_size = len(words)
model = TextCNN(config)
session = tf.Session()
tf.get_variable_scope().reuse_variables()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
print('now ok')
saver.restore(sess=session, save_path=r'cnnModel\data\checkpoints\textcnn\best_validation')


totalCall = 5


def index(request):
    query = 'other'
    if 'query' in request.GET:
        query = request.GET['query']

    x_predict = process_one_strLine(query, word_to_id, max_length=30)
    feed_dict = {
        model.input_x: x_predict,
        model.keep_prob: 1.0
    }
    predict = session.run(model.classIndex, feed_dict=feed_dict)
    classIndex = predict[0]
    result_className = classList[classIndex]

    content = json.dumps( {'domain':result_className,'query':query}, ensure_ascii = False)
    return HttpResponse(content, content_type='application/json; charset=utf-8')

