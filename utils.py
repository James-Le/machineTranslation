__author__ = 'jmh081701'
import  json
import  copy
import  numpy as np
import  random

class  DATAPROCESS:
    def __init__(self,source_ling_path,dest_ling_path,source_word_embedings_path,source_vocb_path,dest_word_embeddings_path,dest_vocb_path,seperate_rate=0.05,batch_size=100):
        self.src_data_path =source_ling_path
        self.dst_data_path =dest_ling_path

        self.src_word_embedding_path = source_word_embedings_path
        self.src_vocb_path  = source_vocb_path

        self.dst_word_embedding_path=dest_word_embeddings_path 
        self.dst_vocb_path = dest_vocb_path    

        self.seperate_rate =seperate_rate       
        self.batch_size = batch_size
        self.src_sentence_length = 32           
        self.dst_sentence_length = 32
        #data structure to build
        self.src_data_raw=[]
        self.dst_data_raw =[]
        self.src_train_raw=[]   
        self.dst_train_raw = []
        self.src_test_raw =[]   
        self.dst_test_raw =[]

        self.src_word_embeddings=None   
        self.src_id2word=None
        self.src_word2id=None
        self.src_embedding_length =0

        self.dst_word_embeddings=None   
        self.dst_id2word=None
        self.dst_word2id=None
        self.dst_embedding_length =0

        self.__load_wordebedding()


        self.__load_data()

        self.last_batch=0
        self.epoch =0
        self.dst_vocb_size = len(self.dst_word2id)

    def __load_wordebedding(self):
        self.src_word_embeddings=np.load(self.src_word_embedding_path)
        self.embedding_length = np.shape(self.src_word_embeddings)[-1]
        with open(self.src_vocb_path,encoding="utf8") as fp:
            self.src_id2word = json.load(fp)
        self.src_word2id={}
        for each in self.src_id2word:
            self.src_word2id.setdefault(self.src_id2word[each],each)

        self.dst_word_embeddings=np.load(self.dst_word_embedding_path)
        self.embedding_length = np.shape(self.dst_word_embeddings)[-1]
        with open(self.dst_vocb_path,encoding="utf8") as fp:
            self.dst_id2word = json.load(fp)
        self.dst_word2id={}
        for each in self.dst_id2word:
            self.dst_word2id.setdefault(self.dst_id2word[each],each)

    def __load_data(self):

        with open(self.src_data_path,encoding='utf8') as fp:
            train_data_rawlines=fp.readlines()
        with open(self.dst_data_path,encoding='utf8') as fp:
            train_label_rawlines=fp.readlines()
        total_lines = len(train_data_rawlines)
        assert len(train_data_rawlines)==len(train_label_rawlines)
        src_len=[]
        dst_len=[]
        for index in range(total_lines):
            data_line = train_data_rawlines[index].split(" ")[:-1]
            label_line = train_label_rawlines[index].split(" ")[:-1]
            label_line =["<s>"]+label_line+["</s>"]    
            #add and seperate valid ,train set.
            data=[int(self.src_word2id.get(each,0)) for each in data_line]
            label=[int(self.dst_word2id.get(each,0)) for each in label_line]
            src_len.append(len(data))
            dst_len.append(len(label))
            self.src_data_raw.append(data)
            self.dst_data_raw.append(label)

            if random.uniform(0,1) <self.seperate_rate:
                self.src_test_raw.append(data)
                self.dst_test_raw.append(label)
            else:
                self.src_train_raw.append(data)
                self.dst_train_raw.append(label)
        self.src_len_std=np.std(src_len)
        self.src_len_mean=np.mean(src_len)
        self.src_len_max=np.max(src_len)
        self.src_len_min=np.min(src_len)

        self.dst_len_std=np.std(dst_len)
        self.dst_len_mean=np.mean(dst_len)
        self.dst_len_max = np.max(dst_len)
        self.dst_len_min=np.min(dst_len)

        self.train_batches= [i for i in range(int(len(self.src_train_raw)/self.batch_size) -1)]
        self.train_batch_index = 0

        self.test_batches= [i for i in range(int(len(self.src_test_raw)/self.batch_size) -1)]
        self.test_batch_index = 0

    def pad_sequence(self,sequence,object_length,pad_value=None):

        sequence =copy.deepcopy(sequence)
        if pad_value is None:
            sequence = sequence*(1+int((0.5+object_length)/(len(sequence))))
            sequence = sequence[:object_length]
        else:
            if len(sequence) < object_length:
                sequence = sequence+[pad_value]*(object_length- len(sequence))
            else:
                sequence = sequence[:object_length-1]+[pad_value]
        return sequence

    def next_train_batch(self):
        #padding
        output_x=[]
        output_label=[]
        src_sequence_length=[]
        dst_sequence_length=[]
        index =self.train_batches[self.train_batch_index]
        self.train_batch_index =(self.train_batch_index +1 ) % len(self.train_batches)
        if self.train_batch_index is 0:
            self.epoch +=1
        datas = self.src_train_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.dst_train_raw[index*self.batch_size:(index+1)*self.batch_size]
        for index in range(self.batch_size):
            #复制填充
            data= self.pad_sequence(datas[index],self.src_sentence_length,pad_value=int(self.src_word2id['</s>']))    
            label = self.pad_sequence(labels[index],self.dst_sentence_length,pad_value=int(self.dst_word2id['</s>'])) 
            label[-1]=int(self.dst_word2id['</s>'])                              
            
            if min(self.src_sentence_length,len(datas[index])) != 0:
                temp1 = data
                temp2 = label
                temp3 = min(self.src_sentence_length,len(datas[index]))
                temp4 = min(self.dst_sentence_length,len(label))
                output_x.append(temp1)
                output_label.append(temp2)
                src_sequence_length.append(temp3)
                dst_sequence_length.append(temp4)
            else:
                try:
                    output_x.append(temp1)
                    output_label.append(temp2)
                    src_sequence_length.append(temp3)
                    dst_sequence_length.append(temp4)
                except:
                    output_x.append(self.pad_sequence(datas[-1],
                                                      self.src_sentence_length,pad_value=int(self.src_word2id['</s>'])))
                    output_label.append(self.pad_sequence(labels[-1],
                                                          self.dst_sentence_length,pad_value=int(self.dst_word2id['</s>'])))
                    src_sequence_length.append(min(self.src_sentence_length,len(datas[-1])))
                    dst_sequence_length.append(min(self.dst_sentence_length,len(labels[-1])))
            
        return output_x,output_label,src_sequence_length,dst_sequence_length

    def next_test_batch(self):
        output_x=[]
        output_label=[]
        src_sequence_length=[]
        dst_sequence_length=[]
        index =self.test_batches[self.test_batch_index]
        self.test_batch_index =(self.test_batch_index +1 ) % len(self.test_batches)
        datas = self.src_test_raw[index*self.batch_size:(index+1)*self.batch_size]
        labels = self.dst_test_raw[index*self.batch_size:(index+1)*self.batch_size]
        for index in range(len(datas)):
            
            data= self.pad_sequence(datas[index],self.src_sentence_length,pad_value=int(self.src_word2id['</s>']))
            label = self.pad_sequence(labels[index],self.dst_sentence_length,pad_value=int(self.dst_word2id['</s>']))
            
            if min(self.src_sentence_length,len(datas[index])) != 0:
                temp1 = data
                temp2 = label
                temp3 = min(self.src_sentence_length,len(datas[index]))
                temp4 = min(self.dst_sentence_length,len(labels[index]))
                output_x.append(temp1)
                output_label.append(temp2)
                src_sequence_length.append(temp3)
                dst_sequence_length.append(temp4)
                
            else:
                try:
                    output_x.append(temp1)
                    output_label.append(temp2)
                    src_sequence_length.append(temp3)
                    dst_sequence_length.append(temp4)
                except:
                    output_x.append(self.pad_sequence(datas[-1],
                                                      self.src_sentence_length,pad_value=int(self.src_word2id['</s>'])))
                    output_label.append(self.pad_sequence(labels[-1],
                                                          self.dst_sentence_length,pad_value=int(self.dst_word2id['</s>'])))
                    src_sequence_length.append(min(self.src_sentence_length,len(datas[-1])))
                    dst_sequence_length.append(min(self.dst_sentence_length,len(labels[-1])))
                    

        return output_x,output_label,src_sequence_length,dst_sequence_length
    def test_data(self):
        output_x=[]
        output_label=[]
        src_sequence_length=[]
        dst_sequence_length=[]
        datas = self.src_test_raw[0:]
        labels = self.dst_test_raw[0:]
#         for index in range(len(datas)):
#             data= self.pad_sequence(datas[index],self.src_sentence_length,pad_value=int(self.src_word2id['<END>']))
#             label = self.pad_sequence(labels[index],self.dst_sentence_length,pad_value=int(self.dst_word2id['<END>']))
#             output_x.append(data)
#             output_label.append(label)
#             src_sequence_length.append(min(self.src_sentence_length,len(datas[index])))
#             dst_sequence_length.append(min(self.dst_sentence_length,len(labels[index])))

        for index in range(self.batch_size):
            #复制填充
            data= self.pad_sequence(datas[index],self.src_sentence_length,pad_value=int(self.src_word2id['</s>']))
            label = self.pad_sequence(labels[index],self.dst_sentence_length,pad_value=int(self.dst_word2id['</s>']))
            
            if min(self.src_sentence_length,len(datas[index])) != 0:
                temp1 = data
                temp2 = label
                temp3 = min(self.src_sentence_length,len(datas[index]))
                temp4 = min(self.dst_sentence_length,len(labels[index]))
                output_x.append(temp1)
                output_label.append(temp2)
                src_sequence_length.append(temp3)
                dst_sequence_length.append(temp4)
                
            else:
                output_x.append(temp1)
                output_label.append(temp2)
                src_sequence_length.append(temp3)
                dst_sequence_length.append(temp4)
        
        start=0
        end=len(datas)
        while len(output_x)< self.batch_size:
            #不满一个batch就填充
            output_x.append(output_x[start])
            output_label.append(output_label[start])
            src_sequence_length.append(src_sequence_length[start])
            dst_sequence_length.append(dst_sequence_length[start])
            start=(start+1) % end
        print(len(output_x))
        return output_x,output_label,src_sequence_length,dst_sequence_length
    def src_id2words(self,ids):
        rst=[]
        for id in ids:
            rst.append(self.src_id2word[str(id)])
        return  " ".join(rst)
    def tgt_id2words(self,ids):
        rst=[]
        for id in ids:
            if id != 0:
                rst.append(self.dst_id2word[str(id)])
            else:
                rst.append("<UNK>")
        return  " ".join(rst)
def evaluate(predict_labels,real_labels,efficient_length):

# predict_labels:[batch_size,sequence_length],real_labels:[batch_size,sequence_length]
    sentence_nums =len(predict_labels)
    predict_cnt=0
    predict_right_cnt=0
    real_cnt=0
    for sentence_index in range(sentence_nums):
        try:
            pass
            #predict_set=extract_named_entity(predict_labels[sentence_index],efficient_length[sentence_index])
            #real_set=extract_named_entity(real_labels[sentence_index],efficient_length[sentence_index])
            #right_=predict_set.intersection(real_set)
            #predict_right_cnt+=len(right_)
            #predict_cnt += len(predict_set)
            #real_cnt +=len(real_set)
        except Exception as exp:
            print(predict_labels[sentence_index])
            print(real_labels[sentence_index])
    precision = predict_right_cnt/(predict_cnt+0.000000000001)
    recall = predict_right_cnt/(real_cnt+0.000000000001)
    F1 = 2 * precision*recall/(precision+recall+0.00000000001)
    return {'precision':precision,'recall':recall,'F1':F1}
