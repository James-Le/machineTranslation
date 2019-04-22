#coding:utf8
from utils import DATAPROCESS
import tensorflow as tf
import numpy as np
batch_size = 300
rnn_size = 200
rnn_num_layers = 1

max_epoch = 350

encoder_embedding_size = 100
decoder_embedding_size = 100
# Learning Rate
lr = 0.001
display_step = 10

# dataGen = DATAPROCESS(source_ling_path="data/cn.txt",
#                           dest_ling_path="data/en.txt",
#                           source_word_embedings_path="data/cn.txt.ebd.npy",
#                           source_vocb_path="data/cn.txt.vab",
#                           dest_word_embeddings_path="data/en.txt.ebd.npy",
#                           dest_vocb_path="data/en.txt.vab",
#                           batch_size=batch_size,
#                           seperate_rate=0.1
#                         )

dataGen = DATAPROCESS(source_ling_path="data/chinese_text_30.txt",
                          dest_ling_path="data/english_text_30.txt",
                          source_word_embedings_path="data/chinese_text_30.txt.ebd.npy",
                          source_vocb_path="data/chinese_text_30.txt.vab",
                          dest_word_embeddings_path="data/english_text_30.txt.ebd.npy",
                          dest_vocb_path="data/english_text_30.txt.vab",
                          batch_size=batch_size,
                          seperate_rate=0.1
                        )

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    source_sequence_len = tf.placeholder(tf.int32, (None,), name="source_sequence_len")
    target_sequence_len = tf.placeholder(tf.int32, (None,), name="target_sequence_len")
    max_target_sequence_len = tf.placeholder(tf.int32, (None,), name="max_target_sequence_len")

    return inputs, targets, learning_rate, source_sequence_len, target_sequence_len, max_target_sequence_len

def encoder_layer(rnn_inputs, rnn_size, rnn_num_layers,
                  source_sequence_len, source_vocab_size, encoder_embedding_size=100):

    encoder_embed = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoder_embedding_size)

    # LSTM cell
    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        return lstm

    lstms = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])
    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(lstms, encoder_embed, source_sequence_len,
                                                        dtype=tf.float32)

    return encoder_outputs, encoder_states

def decoder_layer_inputs(target_data, target_vocab_to_int, batch_size):

    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])

    decoder_inputs = tf.concat([tf.fill([batch_size, 1], int(target_vocab_to_int["<s>"])),
                                ending], 1)

    return decoder_inputs


def decoder_layer_train(encoder_states, decoder_cell, decoder_embed,
                        target_sequence_len, max_target_sequence_len, output_layer,encoder_outputs,source_sequence_len):

    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed,
                                                       sequence_length=target_sequence_len,
                                                       time_major=False)

    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,memory=encoder_outputs,memory_sequence_length=source_sequence_len)

    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,attention_mechanism,attention_layer_size=rnn_size)

    training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
                                                      training_helper,
                                                      decoder_cell.zero_state(batch_size,dtype=tf.float32).clone(cell_state=encoder_states),
                                                      output_layer)

    training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                  impute_finished=True,
                                                                  maximum_iterations=max_target_sequence_len)

    return training_decoder_outputs



def decoder_layer_infer(encoder_states, decoder_cell, decoder_embed, start_id, end_id,
                        max_target_sequence_len, output_layer, batch_size,encoder_outputs,source_sequence_len):

    start_tokens = tf.tile(tf.constant([start_id], dtype=tf.int32), [batch_size], name="start_tokens")
    
    decoder_initial_states = tf.contrib.seq2seq.tile_batch(encoder_states, multiplier=5)

#     inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embed,
#                                                                 start_tokens,
#                                                                 end_id)
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,memory=encoder_outputs,memory_sequence_length=source_sequence_len)
        #加入attention ，加入的方式与train类似
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,attention_mechanism,attention_layer_size=rnn_size)
#     inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,
#                                                        inference_helper,
#                                                        decoder_cell.zero_state(batch_size,dtype=tf.float32).clone(cell_state=encoder_states),
#                                                        output_layer)
    
    # try BeamSearch to improve translation outcomes
    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(decoder_cell,
                                                             decoder_embed,
                                                             start_tokens,
                                                             end_id,
                                                             decoder_initial_states,
                                                             beam_width=5,
                                                             output_layer=output_layer)
    

    inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=max_target_sequence_len)

    return inference_decoder_outputs


def decoder_layer(encoder_states, decoder_inputs, target_sequence_len,
                   max_target_sequence_len, rnn_size, rnn_num_layers,
                   target_vocab_to_int, target_vocab_size, decoder_embedding_size, batch_size,encoder_outputs,source_sequence_length):

    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoder_embedding_size]))
    decoder_embed = tf.nn.embedding_lookup(decoder_embeddings, decoder_inputs)

    def get_lstm(rnn_size):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=456))
        return lstm

    decoder_cell = tf.contrib.rnn.MultiRNNCell([get_lstm(rnn_size) for _ in range(rnn_num_layers)])

    # output_layer logits
    output_layer = tf.layers.Dense(target_vocab_size)

    with tf.variable_scope("decoder"):
        training_logits = decoder_layer_train(encoder_states,
                                               decoder_cell,
                                               decoder_embed,
                                               target_sequence_len,
                                               max_target_sequence_len,
                                               output_layer,
                                               encoder_outputs,source_sequence_length)

    with tf.variable_scope("decoder", reuse=True):
        inference_logits = decoder_layer_infer(encoder_states,
                                               decoder_cell,
                                               decoder_embeddings,
                                               int(target_vocab_to_int["<s>"]),
                                               int(target_vocab_to_int["</s>"]),
                                                max_target_sequence_len,
                                                output_layer,
                                                batch_size,
                                                encoder_outputs,source_sequence_length)

    return training_logits, inference_logits



def seq2seq_model(input_data, target_data, batch_size,
                 source_sequence_len, target_sequence_len, max_target_sentence_len,
                 source_vocab_size, target_vocab_size,
                 encoder_embedding_size, decoder_embeding_size,
                 rnn_size, rnn_num_layers, target_vocab_to_int):
    encoder_outputs, encoder_states = encoder_layer(input_data, rnn_size, rnn_num_layers, source_sequence_len,
                                      source_vocab_size, encoder_embedding_size)

    decoder_inputs = decoder_layer_inputs(target_data, target_vocab_to_int, batch_size)

    training_decoder_outputs, inference_decoder_outputs = decoder_layer(encoder_states,
                                                                       decoder_inputs,
                                                                      target_sequence_len,
                                                                       max_target_sentence_len,
                                                                      rnn_size,
                                                                      rnn_num_layers,
                                                                      target_vocab_to_int,
                                                                      target_vocab_size,
                                                                      decoder_embeding_size,
                                                                       batch_size,encoder_outputs,source_sequence_len)
    return training_decoder_outputs, inference_decoder_outputs


train_graph = tf.Graph()

with train_graph.as_default():
    inputs, targets, learning_rate, source_sequence_len, target_sequence_len, _ = model_inputs()

    max_target_sequence_len = 32
    train_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                  targets,
                                                  batch_size,
                                                  source_sequence_len,
                                                  target_sequence_len,
                                                  max_target_sequence_len,
                                                  len(dataGen.src_word2id),
                                                  len(dataGen.dst_word2id),
                                                  encoder_embedding_size,
                                                  decoder_embedding_size,
                                                  rnn_size,
                                                  rnn_num_layers,
                                                  dataGen.dst_word2id)

    training_logits = tf.identity(train_logits.rnn_output, name="logits")
    inference_logits = tf.identity(inference_logits.sample_id, name="predictions")

    masks = tf.sequence_mask(target_sequence_len, max_target_sequence_len, dtype=tf.float32, name="masks")

    with tf.name_scope("optimization"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(cost)
        clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(clipped_gradients)

f = open('train_loss.txt','w')
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    try:
        loader = tf.train.Saver()
        loader.restore(sess, tf.train.latest_checkpoint('./checkpoints'))
    except Exception as exp:
        print("retrain model")
    saver = tf.train.Saver()
    dataGen.epoch = 1
    
    
    while dataGen.epoch < max_epoch:
            output_x,output_label,src_sequence_length,dst_sequence_length=dataGen.next_train_batch()
#             print(output_x)
#             print(src_sequence_length)
#             for i,x in enumerate(output_x):
#                 if len(x) == 0:
#                     print("0 found!")
#                     print(output_x)
            
#             for i,y in enumerate(src_sequence_length):
#                 if y == 0:
#                     print("0 found!")
#                     print(i)
            
#             for z in dst_sequence_length:
#                 if z == 0:
#                     print("0 found!")
                
            _, loss = sess.run(
                [train_op, cost],
                {inputs: output_x,
                 targets: output_label,
                 learning_rate: lr,
                 source_sequence_len: src_sequence_length,
                 target_sequence_len: dst_sequence_length})
            if dataGen.train_batch_index % display_step == 0 and dataGen.train_batch_index > 0:
                output_x,output_label,src_sequence_length,dst_sequence_length=dataGen.next_test_batch()
                batch_train_logits = sess.run(
                    inference_logits,
                    {inputs: output_x,
                     source_sequence_len: src_sequence_length,
                     target_sequence_len: dst_sequence_length})

                print('Epoch {:>3} - Valid Loss: {:>6.4f}'
                      .format(dataGen.epoch, loss))
                f.write(str(dataGen.epoch)+","+str(loss)+"\n")
                
                if dataGen.epoch % 30 ==0 :
                    
                    saver.save(sess,"checkpoints/dev")
    # Save Model
    saver.save(sess, "checkpoints/dev")
    print('Model Trained and Saved')
f.close()