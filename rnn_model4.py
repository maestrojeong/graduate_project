import sys
import csv
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def list_to_one_hot(array, total_drugs):
    '''
    example
    input:
        total_drugs = total types of drugs
        array = [1,2,...,...]
    return :
        [0,1,1,......]
    index 0 : no drug
    index label 1 to total_drugs : Each drug
    '''
    one_hot = np.zeros(total_drugs+1)
    if len(array)==0:
        return one_hot
    length = len(array)
    for i in range(length):
        one_hot[array[i]]=1
    return one_hot

def get_total_drugs(name):
    '''
    input :
        name  file location
    return :
        return total drug
    '''
    id_drug_seqs = open(name,'r',newline='')
    reader = csv.reader(id_drug_seqs,delimiter=',') 
    for row in reader:
        temp  = int(row[2])
        break
    id_drug_seqs.close()
    return temp

def data_load(name, start = 1, end = 10, minimum_drug_sequence = 5):
    '''
    input : 
        name :  file location
        load the drug_seqs from start to end
        minimum_drug_sequence : remove if the length of sequence is shorter than this value
    return : 
        return the drug seqs
    '''
    id_drug_seqs = open(name,'r',newline='')
    reader = csv.reader(id_drug_seqs,delimiter=',') 
    drug_seqs = [] #drug_seqs => drug_seg => drugs
    index = 0
    for row in reader:
        if index==0:
            index = 1
        elif index<start:
            index+=1
        elif index<end:
            index+=1
            temp = row[1:]
            if len(temp)==0:
                continue
            for i in range(len(temp)):
                temp[i]=int(temp[i])
            drug_seq = []
            drug = []
            for i in range(len(temp)):
                if temp[i]==-1:
                    if i!=0:
                        drug_seq.append(drug)
                    drug=[]
                else:
                    drug.append(temp[i])
            drug_seq.append(drug)
            if len(drug_seq) >= minimum_drug_sequence:
                drug_seqs.append(drug_seq)
        else:
            break
    id_drug_seqs.close()
    return drug_seqs

def data_extractor(name, start,end,weights,biases, parser_size = 10, minimum_drug_sequence = 5):
    drug_seqs = data_load(name, start, end ,minimum_drug_sequence)
    input_data = []
    output_data = []
    for i in range(len(drug_seqs)):
        j=0
        while True:
            if j+parser_size+1 > len(drug_seqs[i]):
                break
            temp_input = []
            for k in range(parser_size):
                temp_input.append(list_to_one_hot(drug_seqs[i][j+k],total_drugs))
            output_data.append(list_to_one_hot(drug_seqs[i][j+parser_size],total_drugs))
            temp_input = np.dot(np.array(temp_input),weights)+biases
            input_data.append(temp_input)
            j+=parser_size+1
    input_data = np.array(input_data)
    decoded_output_data = np.array(output_data)
    output_data = np.dot(decoded_output_data, weights)+ biases
    return {'input' : input_data, 'output' : output_data, 'decoded_output' : decoded_output_data}

def print_spec(total_drugs, year, train_data, test_data):
    print("There are total {} drugs in {}".format(total_drugs, year))
    print("Train data size")
    print("input : {}".format(train_data['input'].shape))
    print("output : {}".format(train_data['output'].shape))
    print("decoded_output : {}".format(train_data['decoded_output'].shape))
    print("Test data size")
    print("input : {}".format(test_data['input'].shape))
    print("output : {}".format(test_data['output'].shape))
    print("decoded_output : {}".format(test_data['decoded_output'].shape))
def recurrent_neural_network(x):
    W1 = tf.Variable(tf.truncated_normal([rnn_size, n_classes],stddev=0.01))
    b1 = tf.Variable(tf.constant(0.01, shape = [n_classes]))

    temp_x = tf.transpose(x, [1,0,2])#crucial
    temp_x = tf.reshape(temp_x, [-1, n_classes])
    temp_x = tf.split(temp_x, num_or_size_splits = parser_size, axis = 0)

    gru_cell = tf.contrib.rnn.GRUCell(rnn_size)
    multi_cell = tf.contrib.rnn.MultiRNNCell([gru_cell]*2) 
    outputs, states = tf.contrib.rnn.static_rnn(multi_cell, temp_x, dtype=tf.float32) 
    temp_outputs = tf.sigmoid(tf.matmul(outputs[-1],W1)+b1)
    return temp_outputs

def print_variables(keys):
    print(keys)
    i = 0
    while True:
        try :
            print(tf.get_collection(keys)[i])
            i+=1
        except IndexError:
            break;

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="year_of_data",type=str)
parser.print_help()
args = parser.parse_args()
year = args.year

file_name = "id_drug_seqs_{}.csv".format(year)

total_drugs = get_total_drugs(file_name) 

#hyper_parameter
parser_size = 10
minimum_drug_sequence = parser_size 
rnn_size = 5
train_index = 1000#600000
K = 30
epoch = 200
batch_size = 10

save_dir = './auto_encoder/save_{}'.format(year)

with tf.Session() as sess: 
    restorer = tf.train.import_meta_graph(os.path.join(save_dir,'auto.meta'))
    restorer.restore(sess, os.path.join(save_dir,'auto'))
    encoder_weights = sess.run(tf.get_collection("trainable_variables")[0]) 
    encoder_biases = sess.run(tf.get_collection("trainable_variables")[1])
    decoder_weights = sess.run(tf.get_collection("trainable_variables")[2])
    decoder_biases = sess.run(tf.get_collection("trainable_variables")[3])
tf.reset_default_graph()
print_variables("variables")
print(encoder_weights.shape)
print(encoder_biases.shape)
print(decoder_weights.shape)
print(decoder_biases.shape)
n_classes = len(encoder_biases)

train_data = data_extractor(file_name, 1,train_index, encoder_weights, encoder_biases, parser_size, minimum_drug_sequence)
test_data = data_extractor(file_name, train_index+1, int(train_index*5/4), encoder_weights, encoder_biases, parser_size, minimum_drug_sequence)

print_spec(total_drugs, year, train_data, test_data)

x = tf.placeholder('float', [None, parser_size, n_classes])
y = tf.placeholder('float', [None, n_classes])

y_hat = recurrent_neural_network(tf.sigmoid(x))
#cost = tf.reduce_mean(tf.square(tf.sigmoid(y) - y_hat))
cost = -tf.reduce_mean(tf.sigmoid(y)*tf.log(tf.clip_by_value(y_hat, clip_value_max = 1-1e-7, clip_value_min = 1e-5))
                        + (1-tf.sigmoid(y))*tf.log(tf.clip_by_value(1-y_hat, clip_value_max = 1-1e-7, clip_value_min = 1e-5)))
train = tf.train.AdamOptimizer(1e-4).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for _ in range(epoch):
    temp_input_data = train_data['input']
    temp_output_data = train_data['output']
    
    shuffle = np.arange(0, len(temp_input_data))
    np.random.shuffle(shuffle)

    input_shuffle_data = []
    output_shuffle_data = []
    
    for j in range(len(temp_input_data)):
        input_shuffle_data.append(temp_input_data[shuffle[j]])
        output_shuffle_data.append(temp_output_data[shuffle[j]])
    
    input_shuffle_data = np.array(input_shuffle_data)
    output_shuffle_data = np.array(output_shuffle_data)
    
    for batch in range(int(len(train_data['input'])/batch_size)): 
        sess.run(train, feed_dict={x: input_shuffle_data[batch*batch_size:(batch+1)*batch_size], 
                                    y: output_shuffle_data[batch*batch_size:(batch+1)*batch_size]})
    if _ %10 == 0:
        print("Epoch({}/{})".format(_+1,epoch))
        print("Train cost = {}".format(sess.run(cost, feed_dict = {x : train_data['input'], y : train_data['output']})))
        print("Test cost = {}".format(sess.run(cost, feed_dict = {x : test_data['input'], y : test_data['output']})))

prediction = np.dot(sess.run(y_hat, feed_dict = {x: test_data['input']}),decoder_weights) + decoder_biases 
total = 0
correct = 0

for i in range(len(prediction)):
    for k in range(total_drugs+1):
        if test_data['decoded_output'][i][k]!=0:
            total+=1
            rank = 0
            for k1 in range(total_drugs+1):
                if prediction[i][k1]>=prediction[i][k]: 
                    rank+=1
            if rank<=K:
                correct+=1
accuracy = correct/total*100.0
f = open('rnn_model4_year{}_hidden{}_datasize{}.txt'.format(year, n_classes,train_index),'w')
f.write("accuracy = {}({}/{})".format(accuracy,correct,total))
f.close()
