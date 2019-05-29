import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import string
from random import shuffle

def seq2seqmodel(x, y, weights, biases):
    tgt_sos_id = hin_ind['#']
    tgt_eos_id = hin_ind['$']
    tgt_pad_id = hin_ind['%']

    x = tf.cast(x, tf.int32)
    y = tf.cast(y, tf.int32)

    inembed = tf.nn.embedding_lookup(weights['inembed'], x)
    #inembed = tf.nn.tanh(inembed)

    enc_lstm_fw = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units = encsize, name = 'basic_lstm_cell'), dropout_keep_prob)
    enc_lstm_bw = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units = encsize, name = 'basic_lstm_cell'), dropout_keep_prob)
    enc_output, (enc_state_fw, enc_state_bw) = tf.nn.bidirectional_dynamic_rnn(enc_lstm_fw, enc_lstm_bw, inembed, dtype=tf.float32)
    enc_output = tf.concat(enc_output, 2)
    enc_state = tf.concat((enc_state_fw.c, enc_state_bw.c), 1)

    decoder_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.BasicRNNCell(decsize), tf.nn.rnn_cell.BasicRNNCell(decsize)]), output_keep_prob = dropout_keep_prob)
    dec_state_1 = enc_state
    dec_state_2 = tf.zeros_like(enc_state)
    dec_state = (dec_state_1, dec_state_2)
    decoder_emb_inp = tf.nn.embedding_lookup(weights['outembed'], y)

    def attention(previous_decoder_state, previous_decoder_output_embedding, annotation_vectors):
        #annotation_vectors is encoder output for all time steps and has dimension [batch_size x max_time_steps x encoder_output_size]
        #previous_decoder_state [batch_size x dec_rnn_size]
        # previous_decoder_output_embedding [batch_size x outembed]
        decoder_input = tf.concat([previous_decoder_state, previous_decoder_output_embedding], axis = 1)

        decoder_score = tf.contrib.layers.fully_connected(decoder_input, 1, activation_fn=tf.nn.tanh)
        decoder_score = tf.squeeze(decoder_score) # [batch_size]

        encoder_scores = tf.contrib.layers.fully_connected(annotation_vectors, 1, activation_fn=tf.nn.tanh)
        encoder_scores = tf.squeeze(encoder_scores) #batch_size x max_encoder_time_steps

        scores = tf.add(encoder_scores, tf.expand_dims(decoder_score, -1)) #uses broadcasting
        attention_weights = tf.nn.softmax(scores)

        ww = tf.broadcast_to(tf.expand_dims(attention_weights,-1), tf.shape(annotation_vectors))
        context_vector = tf.reduce_sum(tf.multiply(annotation_vectors, ww), 1)

        return context_vector, attention_weights

    def loop_fn(time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:
            initial_elements_finished = tf.tile([False], tf.reshape(tf.shape(y)[0], [1]))
            #initial_input, _ = attention(dec_state_1, tf.nn.embedding_lookup(weights['outembed'], tf.reshape(tf.slice(y, [0, time], [tf.shape(y)[0], 1]), tf.reshape(tf.shape(y)[0], [1]))), enc_output)
            #initial_input = tf.nn.embedding_lookup(weights['outembed'], tf.reshape(tf.slice(y, [0, time], [tf.shape(y)[0], 1]), tf.reshape(tf.shape(y)[0], [1])))
            initial_input = tf.concat([tf.zeros([b_size, 2*encsize]) ,tf.nn.embedding_lookup(weights['outembed'], tf.multiply(tf.ones([b_size], dtype=tf.int32, name='SOS'), 2))], axis = 1)
            initial_cell_state = dec_state
            initial_cell_output = tf.zeros([len(hin_alphabet)])
            initial_loop_state = None
            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        elif training == True:
            def get_next_input(output_logits):
                #output_logits = tf.add(tf.matmul(previous_output, weights['softmax']), biases['softmax'])
                prediction = tf.argmax(output_logits, axis=1)
                next_input = tf.nn.embedding_lookup(weights['outembed'], prediction)
                return next_input

            elements_finished = (time >= target_sequence_length)
            
            finished = tf.reduce_all(elements_finished)
            #next_input_ind = tf.cond(finished, lambda: tf.reshape(tf.slice(y, [0, time-1], [tf.shape(y)[0], 1]), tf.reshape(tf.shape(y)[0], [1])), tf.reshape(tf.slice(y, [0, time], [tf.shape(y)[0], 1]), tf.reshape(tf.shape(y)[0], [1])))
            #context_vec, _ = attention(previous_state[0], tf.nn.embedding_lookup(weights['outembed'], next_input_ind), enc_output)
            #next_input = tf.concat([context_vec, tf.nn.embedding_lookup(weights['outembed'], next_input_ind)], axis = 1)
            #next_input = tf.nn.embedding_lookup(weights['outembed'], next_input_ind)
            output = tf.add(tf.matmul(previous_output, weights['softmax']), biases['softmax'])
            next_input = get_next_input(output)
            context_vec, _ = attention(previous_state[0], next_input, enc_output)
            next_input = tf.concat([context_vec, next_input], axis = 1)

            state = previous_state
            loop_state = None

            return (elements_finished, 
                    next_input,
                    state,
                    output,
                    loop_state)

        else:
            def get_next_input(output_logits):
                #output_logits = tf.add(tf.matmul(previous_output, weights['softmax']), biases['softmax'])
                prediction = tf.argmax(output_logits, axis=1)
                next_input = tf.nn.embedding_lookup(weights['outembed'], prediction)
                return next_input
            
            elements_finished = (time >= target_sequence_length)
            
            finished = tf.reduce_all(elements_finished)
            #next_input, _ = attention(previous_state[0], get_next_input(), enc_output)
            #next_input = tf.concat([context_vec, get_next_input()], axis = 1)
            output = tf.add(tf.matmul(previous_output, weights['softmax']), biases['softmax'])
            next_input = get_next_input(output)
            context_vec, _ = attention(previous_state[0], next_input, enc_output)
            next_input = tf.concat([context_vec, next_input], axis = 1)

            state = previous_state
            loop_state = None

            return (elements_finished, 
                    next_input,
                    state,
                    output,
                    loop_state)

    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
    logits = decoder_outputs_ta.stack()
    logits = tf.transpose(logits, [1, 0, 2])

    #logits = tf.reshape(logits, (-1, decsize))
    #logits = tf.add(tf.matmul(logits, weights['softmax']), biases['softmax'])
    #logits = tf.reshape(logits, (b_size, -1, len(hin_alphabet)))

    """
    projection_layer = tf.layers.Dense(len(hin_alphabet), kernel_initializer = initializer, bias_initializer = initializer)
    #attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(decsize, enc_output, memory_sequence_length=source_sequence_length)
    #decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=decsize)
    #dec_state = decoder_cell.zero_state(tf.shape(y)[0], tf.float32).clone(cell_state=dec_state)

    if training==True:
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, target_sequence_length)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_state, output_layer=projection_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished = True, maximum_iterations = max_steps)
        logits = outputs.rnn_output

    else:
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(weights['outembed'], tf.fill([b_size], tgt_sos_id), tgt_eos_id)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_state, output_layer=projection_layer)

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished = True, maximum_iterations = max_steps)
        logits = outputs.rnn_output
    """

    #def decoder_callable(dec_state_1, dec_state_2, i):
    #    dec_output_1, dec_state_1_new = decoder_1(enc_output[:, i, :], dec_state_1)
    #    dec_output_2, dec_state_2_new = decoder_2(dec_output_1, dec_state_2)
    #    softmax_out = tf.nn.softmax(tf.reshape(tf.add(tf.matmul(tf.reshape(dec_output_2, [-1, weights['softmax'].get_shape().as_list()[0]]), weights['softmax']), biases['softmax']), [-1, max_steps, weights['softmax'].get_shape().as_list()[1]]))

    #    outembed = tf.reshape(tf.add(tf.matmul(tf.reshape(softmax_out, [-1, weights['outembed'].get_shape().as_list()[0]]), weights['outembed']), biases['outembed']), [-1, max_steps, weights['outembed'].get_shape().as_list()[1]])
    #    outembed = tf.nn.tanh(outembed)

    #    return (dec_state_1_new, dec_state_2_new, tf.add(i, 1))

    #for_each_time_step = lambda a, b, step: tf.less(step, max_steps)
    #dec_state_1, dec_state_2, _ = tf.while_loop(for_each_time_step, decoder_callable, (dec_state_1, dec_state_2, 0))

    return logits

def removeSpaces(words):
    for i in range(words.shape[0]):
        words[i] = words[i].translate({32:''})

def flatten(x):
    return [item for sublist in x for item in sublist] 

def create_alph_dicts(alphabet):
    return {a: [0 if a!=b else 1 for b in alphabet] for a in alphabet}, {alphabet[i]: i  for i in range(len(alphabet))}

def pad_sequences(x, y, y_in, max_len_x, max_len_y):
    x = [l + [0]*(max_len_x - len(l)) for l in x]
    if y is not None:
        y = [l + [0]*(max_len_y - len(l)) for l in y]
        y_in = [l + [0]*(max_len_y - len(l)) for l in y_in]
    return x, y, y_in

if __name__ == '__main__':
    np.random.seed(1234)
    #using argparse to get parameters according to the problem statement
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default = 0.001)
    parser.add_argument("--batch_size", type=int, help="batch size", default=20)
    parser.add_argument("--init", type=int, choices = [1, 2], help="weight initialization: 1. Xavier, 2. Uniform random", default = 1)
    parser.add_argument("--dropout_prob", type=float, help="dropout probability", default=0.0)
    parser.add_argument("--decode_method", type=int, choices = [0, 1], help="decode method: 0. Greedy, 1. Beam Search", default = 0)
    parser.add_argument("--beam_width", type=int, help="Beam width", default=0)
    parser.add_argument("--save_dir", type=str, help="directory to save model in", default = "models")
    parser.add_argument("--epochs", type=int, help="number of epochs", default=0)
    parser.add_argument("--train", type=str, help="path to training dataset", default="dl2019pa3/train.csv")
    parser.add_argument("--val", type=str, help="path to validation dataset", default="dl2019pa3/valid.csv")
    parser.add_argument("--test", type=str, help="path to test dataset", default="dl2019pa3/partial_test_400.csv")
    parser.add_argument("--infer", type=int, help="model number to make predictions on")
    args = parser.parse_args()

    train_data = pd.read_csv(args.train)
    x_train = train_data.loc[:, 'ENG'].values
    removeSpaces(x_train)
    y_train = train_data.loc[:, 'HIN'].values
    removeSpaces(y_train)

    xy_train = [([a],[b]) if a.count('_') != b.count('_') else (list(filter(None, a.split('_'))), list(filter(None, b.split('_'))))  for a,b in list(set(zip(x_train, y_train)))]
    x_train = flatten([a for a,b in xy_train])
    y_train = flatten([b for a,b in xy_train])

    translation_table = dict.fromkeys(map(ord, ''.join([a for a in set(''.join(x_train)) if a not in string.ascii_uppercase])), None)
    x_train = [a.translate(translation_table) for a in x_train]
    y_train = [a.translate(translation_table) for a in y_train]

    xy_train = [(a, b) for a,b in list(set(zip(x_train, y_train))) if len(a)>0 and len(b)>0]
    x_train = [a for a,b in xy_train]
    y_train = [b for a,b in xy_train]

    eng_alphabet = ['%', '$', '#'] + sorted(set(''.join(x_train)))
    hin_alphabet = ['%', '$', '#'] + sorted(set(''.join(y_train)))

    print(hin_alphabet)

    one_hot_eng, eng_ind = create_alph_dicts(eng_alphabet)
    one_hot_hin, hin_ind = create_alph_dicts(hin_alphabet)

    X_train = [[eng_ind[a] for a in w] for w in x_train]
    Y_train = [[hin_ind[a] for a in w]+[hin_ind['$']] for w in y_train]
    Y_train_inp = [[hin_ind['#']]+[hin_ind[a] for a in w] for w in y_train]

    print('Read training data')

    val_data = pd.read_csv(args.val)
    x_val = val_data.loc[:, 'ENG'].values
    removeSpaces(x_val)
    y_val = val_data.loc[:, 'HIN'].values
    removeSpaces(y_val)
    X_val = [[eng_ind[a] for a in w] for w in x_val]
    Y_val = [[hin_ind[a] for a in w]+[hin_ind['$']] for w in y_val]
    Y_val_inp = [[hin_ind['#']]+[hin_ind[a] for a in w] for w in y_val]
    print('Read validation data')

    test_data = pd.read_csv(args.test)
    x_test = test_data.loc[:, 'ENG'].values
    removeSpaces(x_test)
    X_test = [[eng_ind[a] for a in w] for w in x_test]
    print('Read test data')

    if args.init == 1:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = None

    encsize = 256
    decsize = 512
    outembsize = 256

    weights = {
        'inembed': tf.get_variable('W0', shape=(len(eng_alphabet), encsize), initializer=initializer),
        'softmax': tf.get_variable('W1', shape=(decsize, len(hin_alphabet)), initializer=initializer),
        'outembed': tf.get_variable('W2', shape=(len(hin_alphabet), outembsize), initializer=initializer),
    }

    biases = {
        'inembed': tf.get_variable('B0', shape=(encsize,), initializer=initializer),
        'softmax': tf.get_variable('B1', shape=(len(hin_alphabet),), initializer=initializer),
        'outembed': tf.get_variable('B2', shape=(outembsize,), initializer=initializer),
    }

    x = tf.placeholder("float", [None, None])
    y_out = tf.placeholder("float", [None, None])
    y = tf.placeholder("float", [None, None])
    target_sequence_length = tf.placeholder(tf.int32, [None])
    source_sequence_length = tf.placeholder(tf.int32, [None])
    max_steps = tf.placeholder(tf.int32, shape = ())
    training = tf.placeholder("bool", None)
    b_size = tf.placeholder(tf.int32, shape = ())
    dropout_keep_prob = tf.placeholder("float", None)

    pred_nopad = seq2seqmodel(x, y, weights, biases)

    #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.cast(y_out, tf.int32)))

    masks = tf.sequence_mask(target_sequence_length, tf.reduce_max(target_sequence_length), dtype=tf.float32, name='masks')
    pad_size = tf.reshape(tf.shape(y_out)[1] - tf.shape(pred_nopad)[1], [1])
    pred = tf.pad(pred_nopad, tf.stack([tf.constant([0, 0]),tf.concat([tf.constant([0]), pad_size], axis = 0),tf.constant([0, 0])]))
    cost = tf.contrib.seq2seq.sequence_loss(pred, tf.cast(y_out, tf.int32), masks)

    params = tf.trainable_variables()
    gradients = tf.gradients(cost, params)
    max_gradient_norm = 1
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
    update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

    correct_prediction = tf.reduce_all(tf.equal(tf.argmax(pred, 2), tf.cast(y_out, tf.int64)), axis = 1)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep = 50)
    
    with tf.Session() as sess:
        if args.infer is not None:
            saver.restore(sess, "models/model{}.ckpt".format(args.infer))
            max_len_x = len(max(X_test, key = len))
            max_len_y = 100
            in_sequence_lengths = list(map(len, X_test))
            out_sequence_lengths = [max_len_y]*len(X_test)
            batch_x, _, _ = pad_sequences(X_test[:], None, None, max_len_x, None)

            prediction = sess.run(pred_nopad, feed_dict={x: batch_x, y: batch_x, y_out: batch_x, training: False, b_size: len(batch_x), max_steps: max_len_y, dropout_keep_prob: 1.0, source_sequence_length: in_sequence_lengths, target_sequence_length: out_sequence_lengths})
            print(prediction)
            output = np.argmax(prediction, 2)
            print(hin_ind)
            print(output)
            output = [(' '.join([hin_alphabet[i] for i in l])).split(' $')[0] for l in output]
            df = pd.DataFrame(output, columns = ['HIN'])
            print(df)
            df.to_csv('prediction.csv', index_label = 'id')
        else:
            sess.run(init)
            summary_writer = tf.summary.FileWriter('./Output', sess.graph)
            for i in range(args.epochs):
                XY_train = list(zip(X_train, Y_train, Y_train_inp))
                shuffle(XY_train)
                X_train, Y_train, Y_train_inp = zip(*XY_train)

                total_num_correct = 0.0
                total_loss = 0.0
                for batch in range(((len(x_train)-1)//args.batch_size)+1):
                    if batch%10==0:
                        print(batch)
                    batch_x = X_train[batch*args.batch_size:min((batch+1)*args.batch_size,len(x_train))]
                    batch_y_inp = Y_train_inp[batch*args.batch_size:min((batch+1)*args.batch_size,len(y_train))]
                    batch_y = Y_train[batch*args.batch_size:min((batch+1)*args.batch_size,len(y_train))]
                    max_len_x = len(max(batch_x, key = len))
                    max_len_y = len(max(batch_y, key = len))
                    in_sequence_lengths = list(map(len, batch_x))
                    out_sequence_lengths = list(map(len, batch_y))
                    batch_x, batch_y, batch_y_inp = pad_sequences(batch_x, batch_y, batch_y_inp, max_len_x, max_len_y)

                    #print(max_len_y, batch_x, batch_y, batch_y_inp)
                    #out, psize = sess.run([pred, pad_size], feed_dict={x: batch_x, y: batch_y_inp, y_out: batch_y, training: False, b_size: len(batch_x), max_steps: max_len_y, dropout_keep_prob: 1.0, target_sequence_length: sequence_lengths})
                    #print(np.array(out).shape, np.argmax(out, 2), psize)
                    
                    opt = sess.run(update_step, feed_dict={x: batch_x, y: batch_y_inp, y_out: batch_y, training: True, b_size: len(batch_x), max_steps: max_len_y, dropout_keep_prob: (1.0-args.dropout_prob), target_sequence_length: out_sequence_lengths, source_sequence_length: in_sequence_lengths})
                    loss, n_corr, acc = sess.run([cost, num_correct, accuracy], feed_dict={x: batch_x, y: batch_y_inp, y_out: batch_y, training: False, b_size: len(batch_x), max_steps: max_len_y, dropout_keep_prob: 1.0, target_sequence_length: out_sequence_lengths, source_sequence_length: in_sequence_lengths})
                    #print(acc)
                    total_loss += (loss-total_loss)/(batch+1.0)
                    total_num_correct += n_corr
                print("Iter " + str(i) + ", Loss=", "{:.5f}".format(total_loss) , ", Training Accuracy=", "{:.5f}".format(total_num_correct/len(X_train)))
                print("Optimization Finished!")

                batch_x = X_val[:]
                batch_y_inp = Y_val_inp[:]
                batch_y = Y_val[:]
                max_len_x = len(max(batch_x, key = len))
                max_len_y = len(max(batch_y, key = len))
                in_sequence_lengths = list(map(len, batch_x))
                out_sequence_lengths = list(map(len, batch_y))
                batch_x, batch_y, batch_y_inp = pad_sequences(batch_x, batch_y, batch_y_inp, max_len_x, max_len_y)

                loss, n_corr, acc = sess.run([cost, num_correct, accuracy], feed_dict={x: batch_x, y: batch_y_inp, y_out: batch_y, training: False, b_size: len(batch_x), max_steps: max_len_y, dropout_keep_prob: 1.0, target_sequence_length: out_sequence_lengths, source_sequence_length: in_sequence_lengths})
                out = sess.run(pred, feed_dict={x: batch_x, y: batch_y_inp, y_out: batch_y, training: False, b_size: len(batch_x), max_steps: max_len_y, dropout_keep_prob: 1.0, target_sequence_length: out_sequence_lengths, source_sequence_length: in_sequence_lengths})
                print("Validation Loss:","{:.6f}".format(loss),", Validation Accuracy:","{:.5f}".format(acc))
                
                #if i%5==0:
                #    if i!=0 and prev_valid_loss<valid_loss:
                #        print('Early stopping... Best epoch is at epoch', i-5)
                #        break
                #    prev_valid_loss = valid_loss

                save_path = saver.save(sess, args.save_dir+"/model{}.ckpt".format(i))
            summary_writer.close()