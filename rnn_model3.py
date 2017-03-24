import sys
import csv
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

def data_extractor(name, start,end, parser_size = 10, minimum_drug_sequence = 5, train_ratio = 0.8):
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
            temp_input = np.array(temp_input)
            input_data.append(temp_input)
            j+=parser_size+1
    

    train = {}
    train['input']=[]
    train['output']=[]
    test = {}
    test['input']=[]
    test['output']=[]
    
    for i in range(len(input_data)):
        if np.random.random()<train_ratio:
            train['input'].append(input_data[i])
            train['output'].append(output_data[i])
        else:
            test['input'].append(input_data[i])
            test['output'].append(output_data[i])
    train['input'] = np.array(train['input'])
    train['output'] = np.array(train['output'])
    test['input'] = np.array(test['input'])
    test['output'] = np.array(test['output'])
    return {'train' : train, 'test' : test}

def print_spec(total_drugs, year, train_data, test_data):
    print("There are total {} drugs in {}".format(total_drugs, year))
    print("Train data size")
    print("input : {}".format(train_data['input'].shape))
    print("output : {}".format(train_data['output'].shape))
    print("Test data size")
    print("input : {}".format(test_data['input'].shape))
    print("output : {}".format(test_data['output'].shape))

def recurrent_neural_network(x):
    W1 = tf.Variable(tf.truncated_normal([rnn_size, n_classes],stddev=0.01))
    b1 = tf.Variable(tf.constant(0.01, shape = [n_classes]))

    temp_x = tf.transpose(x, [1,0,2])#crucial
    temp_x = tf.reshape(temp_x, [-1, n_classes])
    temp_x = tf.split(temp_x, num_or_size_splits = parser_size, axis = 0)

    gru_cell = tf.contrib.rnn.GRUCell(rnn_size)
    drop_out_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, input_keep_prob = 1.0, output_keep_prob = 0.7)
#    multi_cell = tf.contrib.rnn.MultiRNNCell([drop_out_cell]*2) 
    outputs, states = tf.contrib.rnn.static_rnn(drop_out_cell, temp_x, dtype=tf.float32) 
    temp_outputs = tf.sigmoid(tf.matmul(outputs[-1],W1)+b1)
    return temp_outputs

parser = argparse.ArgumentParser()
parser.add_argument("--year", help="year_of_data",type=str)
parser.print_help()
args = parser.parse_args()
year = args.year

file_name = "id_drug_seqs_{}.csv".format(year)

total_drugs = get_total_drugs(file_name) 
n_classes = total_drugs+1

#hyper_parameter
parser_size = 10
minimum_drug_sequence = parser_size 
rnn_size = 256
index = 830000#60000
epoch = 100
batch_size = 100

result = data_extractor(file_name, 1, index, parser_size, minimum_drug_sequence, 0.8)

train_data = result['train']
test_data = result['test']

print_spec(total_drugs, year, train_data, test_data)

x = tf.placeholder('float', [None, parser_size, n_classes])
y = tf.placeholder('float', [None, n_classes])

y_hat = recurrent_neural_network(x)
cost = -tf.reduce_mean(100*y*tf.log(tf.clip_by_value(y_hat, clip_value_max = 1-1e-7, clip_value_min = 1e-7))
                        + (1-y)*tf.log(tf.clip_by_value(1-y_hat, clip_value_max = 1-1e-7, clip_value_min = 1e-7)))
train = tf.train.AdamOptimizer(1e-2).minimize(cost)

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

    for i in range(len(temp_input_data)):
        input_shuffle_data.append(temp_input_data[shuffle[i]])
        output_shuffle_data.append(temp_output_data[shuffle[i]])
    
    input_shuffle_data = np.array(input_shuffle_data)
    output_shuffle_data = np.array(output_shuffle_data)
   
    for batch in range(int(len(train_data['input'])/batch_size)): 
        sess.run(train, feed_dict={x: input_shuffle_data[batch*batch_size:(batch+1)*batch_size], 
                                    y: output_shuffle_data[batch*batch_size:(batch+1)*batch_size]})
    if _ %10 == 0:
        print("Epoch({}/{})".format(_+1,epoch))
        print("Train cost = {}".format(sess.run(cost, feed_dict = {x : train_data['input'], y : train_data['output']})))
        print("Test cost = {}".format(sess.run(cost, feed_dict = {x : test_data['input'], y : test_data['output']})))

prediction = sess.run(y_hat, feed_dict = {x: test_data['input']})

f = open('rnn_model3_year{}_datasize{}.txt'.format(year, index),'w')
total = 0
correct = 0
K = 5

for i in range(len(prediction)):
    for k in range(total_drugs+1):
        if test_data['output'][i][k]!=0:
            total+=1
            rank = 0
            for k1 in range(total_drugs+1):
                if prediction[i][k1]>=prediction[i][k]: 
                    rank+=1
            if rank<=K:
                correct+=1
accuracy = correct/total*100.0
f.write("{} Value accuracy = {}({}/{})\n".format(K,accuracy,correct,total))

total = 0
correct = 0
K = 10

for i in range(len(prediction)):
    for k in range(total_drugs+1):
        if test_data['output'][i][k]!=0:
            total+=1
            rank = 0
            for k1 in range(total_drugs+1):
                if prediction[i][k1]>=prediction[i][k]: 
                    rank+=1
            if rank<=K:
                correct+=1
accuracy = correct/total*100.0
f.write("{} Value accuracy = {}({}/{})\n".format(K,accuracy,correct,total))

total = 0
correct = 0
K = 20

for i in range(len(prediction)):
    for k in range(total_drugs+1):
        if test_data['output'][i][k]!=0:
            total+=1
            rank = 0
            for k1 in range(total_drugs+1):
                if prediction[i][k1]>=prediction[i][k]: 
                    rank+=1
            if rank<=K:
                correct+=1
accuracy = correct/total*100.0
f.write("{} Value accuracy = {}({}/{})\n".format(K,accuracy,correct,total))

total = 0
correct = 0
K = 30

for i in range(len(prediction)):
    for k in range(total_drugs+1):
        if test_data['output'][i][k]!=0:
            total+=1
            rank = 0
            for k1 in range(total_drugs+1):
                if prediction[i][k1]>=prediction[i][k]: 
                    rank+=1
            if rank<=K:
                correct+=1
accuracy = correct/total*100.0

f.write("{} Value accuracy = {}({}/{})\n".format(K,accuracy,correct,total))

total = 0
correct = 0
K = 40

for i in range(len(prediction)):
    for k in range(total_drugs+1):
        if test_data['output'][i][k]!=0:
            total+=1
            rank = 0
            for k1 in range(total_drugs+1):
                if prediction[i][k1]>=prediction[i][k]: 
                    rank+=1
            if rank<=K:
                correct+=1
accuracy = correct/total*100.0
f.write("{} Value accuracy = {}({}/{})\n".format(K,accuracy,correct,total))
f.close()
