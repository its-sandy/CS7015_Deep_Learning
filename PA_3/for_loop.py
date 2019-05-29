import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import string

def removeSpaces(words):
    for i in range(words.shape[0]):
        words[i] = words[i].translate({32:''})

def flatten(x):
    return [item for sublist in x for item in sublist] 

def create_alph_dicts(alphabet):
    return {alphabet[i]: i  for i in range(len(alphabet))}, {i:alphabet[i] for i in range(len(alphabet))}

def encoder(embed, rnn_size = 256):
    cell_forward = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(rnn_size), dropout_keep_prob)
    cell_backward = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(rnn_size), dropout_keep_prob)
    ((enc_outputs_fw, enc_outputs_bw), (enc_final_state_fw, enc_final_state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, embed, dtype = tf.float32)

    enc_outputs = tf.concat((enc_outputs_fw, enc_outputs_bw), 2)
    enc_final_state_h = tf.concat((enc_final_state_fw.h, enc_final_state_bw.h), 1)
    enc_final_state_c = tf.concat((enc_final_state_fw.c, enc_final_state_bw.c), 1)
    enc_final_state = tf.nn.rnn_cell.LSTMStateTuple(enc_final_state_c, enc_final_state_h)

    return enc_outputs, enc_final_state

def attention(previous_decoder_state, previous_decoder_output_embedding, annotation_vectors):
    #annotation_vectors is encoder output for all time steps and has dimension [batch_size x max_time_steps x encoder_output_size]
    #previous_decoder_state [batch_size x dec_rnn_size]
    # previous_decoder_output_embedding [batch_size x outembed]
    decoder_input = tf.concat([previous_decoder_state, previous_decoder_output_embedding], axis = 1)

    decoder_score = tf.contrib.layers.fully_connected(decoder_input, 1)
    decoder_score = tf.squeeze(decoder_score) # [batch_size]

    encoder_scores = tf.contrib.layers.fully_connected(annotation_vectors, 1)
    encoder_scores = tf.squeeze(encoder_scores) #batch_size x max_encoder_time_steps

    scores = tf.add(encoder_scores, tf.expand_dims(decoder_score, -1)) #uses broadcasting
    attention_weights = tf.nn.softmax(scores)

    ww = tf.broadcast_to(tf.expand_dims(attention_weights,-1), tf.shape(annotation_vectors))
    context_vector = tf.reduce_sum(tf.multiply(annotation_vectors, ww), 1)

    return context_vector, attention_weights


def decoder(context_vector, decoder_input, prev_state1, prev_state2, target_vocab_size, rnn_size = 512):
    #context_vector -> [batch_size x encoder_output_size]
    #decoder_input -> [batch_size x outembed]
    #prev_states, prev_states -> [batch_size x decoder_state_size]

    decoder_full_input = tf.concat([context_vector, decoder_input], 1)
    cell_layer1 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(rnn_size), dropout_keep_prob)
    cell_layer2 = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(rnn_size), dropout_keep_prob)

    output1, state1 = cell_layer1(decoder_full_input, prev_state1)
    output2, state2 = cell_layer2(output1, prev_state2)

    logit = tf.contrib.layers.fully_connected(output2, target_vocab_size, activation_fn=tf.nn.tanh)

    return state1, state2, logit

def seq2seq(source_vocab_size, target_vocab_size, x, y = None, enc_rnn_size=256, dec_rnn_size=512, inembed=256, outembed=256):
    #both x and y should be given as a sequence of integers
    batch_size = tf.shape(x)[0]
    max_encoder_steps = tf.shape(x)[1]
    if istraining == True:
        max_decoder_steps = tf.shape(y)[1]
    else:
        max_decoder_steps = 50

    embedded_inputs = tf.contrib.layers.embed_sequence(x, vocab_size=source_vocab_size, embed_dim=inembed)
    encoder_outputs, encoder_final_state = encoder(embedded_inputs) #batch_size x max_encoder_steps x (2*enc_rnn_size)

    dec_state1 = encoder_final_state
    # dec_state2 = tf.nn.rnn_cell.LSTMStateTuple(tf.zeros([batch_size, dec_rnn_size], tf.float32), tf.zeros([batch_size, dec_rnn_size], tf.float32))
    dec_state2 = encoder_final_state
    embedded_output = tf.zeros([batch_size, outembed], tf.float32)

    logits = []
    preds = []
    for step in range(max_decoder_steps):
        context_vector, _ = attention(dec_state1.h, embedded_output, encoder_outputs)
        dec_state1, dec_state2, logit = decoder(context_vector, embedded_output, dec_state1, dec_state2, target_vocab_size)
        logits.append(logit)
        pred = tf.argmax(logit, 1) #row vector
        preds.append(pred)
        if istraining == True:
            correct_pred = y[:,step]
        else:
            correct_pred = pred
        embedded_output = tf.contrib.layers.embed_sequence(correct_pred, vocab_size=target_vocab_size, embed_dim=outembed)
    
    preds = tf.transpose(tf.stack(preds))
    logits = tf.stack(logits)
    return logits, preds

if __name__ == '__main__':
    #plurals are used in names of tensors only if they span over multiple time steps (ansd not multiple samples in batch size)
    np.random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default = 0.001)
    parser.add_argument("--dropout_keep_prob", type=float, help="dropout keep probability", default = 0.5)
    parser.add_argument("--batch_size", type=int, help="batch size", default=20)
    parser.add_argument("--save_dir", type=str, help="directory to save model in", default = "models")
    parser.add_argument("--epochs", type=int, help="number of epochs", default=0)
    parser.add_argument("--train", type=str, help="path to training dataset", default="dl2019pa3/train.csv")
    parser.add_argument("--val", type=str, help="path to validation dataset", default="dl2019pa3/valid.csv")
    parser.add_argument("--test", type=str, help="path to test dataset", default="dl2019pa3/partial_test_400.csv")
    args = parser.parse_args()

    train_data = pd.read_csv(args.train)
    x_train = train_data.loc[:, 'ENG'].values
    removeSpaces(x_train)
    y_train = train_data.loc[:, 'HIN'].values
    removeSpaces(y_train)

    xy_train = [([a],[b]) if a.count('_') != b.count('_') else (filter(None, a.split('_')), filter(None, b.split('_')))  for a,b in list(set(zip(x_train, y_train)))]
    x_train = flatten([a for a,b in xy_train])
    y_train = flatten([b for a,b in xy_train])

    translation_table = dict.fromkeys(map(ord, ''.join([a for a in set(''.join(x_train)) if a not in string.ascii_uppercase])), None)
    x_train = [a.translate(translation_table) for a in x_train]
    y_train = [a.translate(translation_table) for a in y_train]

    eng_alphabet = ['#', '$', '@'] + list(set(''.join(x_train)))
    hin_alphabet = ['#', '$', '@'] + list(set(''.join(y_train)))

    eng_ind, ind_eng = create_alph_dicts(eng_alphabet)
    hin_ind, ind_hin = create_alph_dicts(hin_alphabet)
    print(eng_ind)
    print(hin_ind)

    x_train = [('#' + w + '$').ljust(50, '@') for w in x_train]
    y_train = [('#' + w + '$').ljust(50, '@') for w in y_train]

    X_train = np.array([[eng_ind[a] for a in w] for w in x_train])
    Y_train = np.array([[hin_ind[a] for a in w] for w in y_train])
    print('Read training data')

    val_data = pd.read_csv(args.val)
    x_val = val_data.loc[:, 'ENG'].values
    removeSpaces(x_val)
    y_val = val_data.loc[:, 'HIN'].values
    removeSpaces(y_val)

    x_val = [('#' + w + '$').ljust(50, '@') for w in x_val]
    y_val = [('#' + w + '$').ljust(50, '@') for w in y_val]

    X_val = np.array([[eng_ind[a] for a in w] for w in x_val])
    Y_val = np.array([[hin_ind[a] for a in w] for w in y_val])
    print('Read validation data')

    test_data = pd.read_csv(args.test)
    x_test = test_data.loc[:, 'ENG'].values
    removeSpaces(x_test)

    x_test = [('#' + w + '$').ljust(50, '@') for w in x_test]
    
    X_test = np.array([[eng_ind[a] for a in w] for w in x_test])
    print('Read test data')
    print("X_train", np.shape(X_train), "Y_train", np.shape(Y_train), "X_val", np.shape(X_val), "Y_val", np.shape(Y_val), "X_test", np.shape(X_test))
    
    #############################3

    x = tf.placeholder(tf.int32, [None, 50])
    y = tf.placeholder(tf.int32, [None, 50])
    dropout_keep_prob = tf.placeholder(tf.float32, None)
    istraining = tf.placeholder(tf.bool, None)
    y_onehot = tf.one_hot(y, len(hin_alphabet))
    logits, preds = seq2seq(source_vocab_size=len(eng_alphabet), target_vocab_size=len(hin_alphabet), x=x, y=y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_onehot))

    
    optimizer = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(cost)
    correct_prediction = tf.reduce_all(tf.equal(tf.cast(preds, tf.int32), y), 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep = 50)

    with tf.Session() as sess:
        """
        saver.restore(sess, "models/model71.ckpt")
        prediction = sess.run([preds], feed_dict={x: X_test, y: None, dropout_keep_prob: 1.0, istraining: False})
        print(prediction)
        output = np.argmax(prediction[0], 1)
        print(output)
        df = pd.DataFrame(output, columns = ['label'])
        print(df)
        df.to_csv('prediction.csv', index_label = 'id')
        """
        sess.run(init)
        summary_writer = tf.summary.FileWriter('./Output', sess.graph)
        for i in range(args.epochs):
            perm = np.random.permutation(len(X_train))
            X_train = X_train[perm]
            Y_train = Y_train[perm]

            total_num_correct = 0.0
            total_loss = 0.0
            for batch in range(((len(X_train)-1)//args.batch_size)+1):
                if batch%10==0:
                    print(batch)
                batch_x = X_train[batch*args.batch_size:min((batch+1)*args.batch_size,len(X_train))]
                batch_y = Y_train[batch*args.batch_size:min((batch+1)*args.batch_size,len(Y_train))]
                
                opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, dropout_keep_prob: args.dropout_keep_prob, istraining: True})
                # loss, n_corr = sess.run([cost, num_correct], feed_dict={x: batch_x, y: batch_y, dropout_keep_prob: 1.0, istraining: False})
                # total_loss += loss
                # total_num_correct += n_corr

            train_acc, train_loss = sess.run([accuracy,cost], feed_dict={x: X_train, y: Y_train, dropout_keep_prob: 1.0, istraining: False})
            val_acc, val_loss, val_preds = sess.run([accuracy,cost,preds], feed_dict={x: X_val, y: Y_val, dropout_keep_prob: 1.0, istraining: False})
            print("Iter " + str(i) + ": ")
            print("Train Loss=", "{:.5f}".format(train_loss) , ", Training Accuracy=", "{:.5f}".format(train_acc))            
            print("Validation Loss:","{:.5f}".format(val_loss),", Validation Accuracy:","{:.5f}".format(val_acc))

            val_preds = [''.join([ind_hin[w] for w in word]) for word in val_preds]
            print(val_preds)
            input('press enter')
            print(Y_val)
            # save_path = saver.save(sess, args.save_dir+"/model{}.ckpt".format(i))
        summary_writer.close()
