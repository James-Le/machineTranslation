__author__ = 'jmh081701'
import  tensorflow as tf
from  utils import  DATAPROCESS

dataGen = DATAPROCESS(source_ling_path="data/chinese_text_30.txt",
                          dest_ling_path="data/english_text_30.txt",
                          source_word_embedings_path="data/chinese_text_30.txt.ebd.npy",
                          source_vocb_path="data/chinese_text_30.txt.vab",
                          dest_word_embeddings_path="data/english_text_30.txt.ebd.npy",
                          dest_vocb_path="data/english_text_30.txt.vab",
                          batch_size=300,
                          seperate_rate=1,
                        )
            #所有的test里面的样本都拿去测试,seperate_rate 于是应该是100%,表示所有的样本都分离开来了



loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph('checkpoints/dev.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./checkpoints'))

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_len:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_len:0')
    print("inference begin ")
    output_x,output_label,src_sequence_length,dst_sequence_length=dataGen.test_data()
    print("inference")

    translate_logits=sess.run(fetches=logits,feed_dict={input_data:output_x,target_sequence_length:dst_sequence_length,source_sequence_length:src_sequence_length})
    for i in range(300):
        src=dataGen.src_id2words(output_x[i])
        dst=dataGen.tgt_id2words(translate_logits[i])
        print({"src":src})
        print({'dst':dst})
        print("Next Line")
