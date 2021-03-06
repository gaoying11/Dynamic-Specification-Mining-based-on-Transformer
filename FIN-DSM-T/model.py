import tensorflow as tf
from tensorflow.python.ops import rnn_cell
#from tensorflow.python.ops import seq2seq
import random
import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                # inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                # inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
                # embedding = tf.compat.v1.get_variable("embedding",
                #                                       [args.vocab_size, args.rnn_size])  # [方法数,32]把方法转化为32位向量 组成矩阵形式
                # tf.split(准备切分的张量，切成几份，在第几个维度上切分)把一个张量划分为几个子张量，0第一个维度，1第二个维度
                # tf.nn.embedding_look(tensor,id)选取一个张量embedding中索引self.input_data对应的元素
                # 比如input_ids=[1,3,5]，则找出embeddings中第1，3，5行，组成一个tensor返回
                # 将input_data用embedding表示
                inputs = tf.nn.embedding_lookup(embedding,
                                                self.input_data)  # 把输入数据以向量形式表示[batch_size, seq_length, rnn_size][10,25,32]
                inputs = tf.split(inputs, args.seq_length, 1)  # 按seq_length划分表示一个步骤的输入[batch_size, 1, rnn_size][10,1,32]
                # inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                # print(inputs)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]  # [10,32]
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        #seq2seq.rnn_decoder
        outputs, last_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        self.saved_outputs=outputs
        output = tf.reshape(tf.concat( outputs,1), [-1, args.rnn_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=200, prime='first all', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        if not len(prime) or prime == " ":
            prime  = random.choice(list(vocab.keys()))    
        print (prime)
        for word in prime.split()[:-1]:
            print (word)
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word,0)
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)
         
        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        word = prime.split()[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab.get(word,0)
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if word == '\n':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = words[sample]
            ret += ' ' + pred
            word = pred
        return ret


