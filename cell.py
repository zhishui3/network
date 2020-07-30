import tensorflow as tf
from tensorflow.contrib import rnn


class BiDirLstmEncoder:
    def __init__(self, args):
        self.args = args

    def __call__(self, x, mask, dropout_keep_prob, name='encoder'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # 定义一个两层的GRU模型
            x = mask * x
            seq_len = tf.count_nonzero(mask, axis=[1, 2])
            lstm_fw = rnn.LSTMCell(num_units=self.args.hidden_size,dtype=tf.float32)
            lstm_bw = rnn.LSTMCell(num_units=self.args.hidden_size,dtype=tf.float32)
            (output_fw, output_bw),_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw,lstm_bw,x,sequence_length=seq_len,dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=2)
            '''
            bidirlstm_layers = rnn.MultiRNNCell(
                [bidirectlstm for i in range(self.args.enc_layers)],
                state_is_tuple=True)
            '''
            #x, h_ = tf.nn.dynamic_rnn(bidirectlstm, x, dtype=tf.float32, sequence_length=seq_len)
            return output
           
class GRU_Encoder:
    def __init__(self, args):
        self.args = args

    def __call__(self, x, mask, dropout_keep_prob, name='encoder'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # 定义一个两层的GRU模型
            x = mask * x
            seq_len = tf.count_nonzero(mask, axis=[1, 2])
            gru_layers = rnn.MultiRNNCell([rnn.GRUCell(num_units=self.args.hidden_size) for i in range(self.args.enc_layers)], state_is_tuple=True)
            x, h_ = tf.nn.dynamic_rnn(gru_layers, x, dtype=tf.float32, sequence_length=seq_len)
            return x
           
           
