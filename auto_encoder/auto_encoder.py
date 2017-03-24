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

def sigmoid_linear(input, output_dim):
    W = tf.get_variable("weights", [input.get_shape()[1], output_dim], initializer=tf.random_normal_initializer())
    b = tf.get_variable("biases", [output_dim], initializer=tf.constant_initializer(0.0))    
    return tf.sigmoid(tf.matmul(input, W) + b)

def print_variables(keys):
    print(keys)
    i = 0
    while True:
        try :
            print(tf.get_collection(keys)[i])
            i+=1
        except IndexError:
            break;

def create_save_dir(year):
    directory = "./save_{}".format(year)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="year_of_data",type=str)
args = parser.parse_args()
year = args.year

save_dir = create_save_dir(year)
file_name = "/home/guest/clinic/yeonwoo/id_drug_seqs_{}.csv".format(year)
total_drugs = get_total_drugs(file_name)
print("There are total {} drugs in {}".format(total_drugs, year))

train_steps = 2000
hidden_units = 15
start = 1
end = 100000
batch_size = 100
epoch = 10

drug_seqs = data_load(file_name, start, end, 1)
input_data = []
for i in range(len(drug_seqs)):
    for j in range(len(drug_seqs[i])):    
        input_data.append(list_to_one_hot(drug_seqs[i][j], total_drugs))

input_data = np.array(input_data)
print("Train input size : {}".format(input_data.shape))

x = tf.placeholder(tf.float32,[None, total_drugs+1],"encoder_input")

with tf.variable_scope("encoder"):
    hidden_layer = sigmoid_linear(x, hidden_units) 
with tf.variable_scope("decoder") as scope:
    x_hat = sigmoid_linear(hidden_layer, total_drugs+1)

cost = -tf.reduce_mean(x*tf.log(tf.clip_by_value(x_hat, clip_value_max = 1-1e-5, clip_value_min = 1e-5))
                        + (1-x)*tf.log(tf.clip_by_value(1-x_hat, clip_value_max = 1-1e-5, clip_value_min = 1e-5)))

train = tf.train.AdamOptimizer(1e-3).minimize(cost)
print_variables("variables")
print_variables("trainable_variables")

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(epoch):
    batch_number = int(len(input_data)/batch_size)
    shuffle = np.arange(0,len(input_data))
    np.random.shuffle(shuffle)
    input_shuffle = []
    for j in range(len(input_data)):
        input_shuffle.append(input_data[shuffle[j]])
    input_shuffle = np.array(input_shuffle)
    for j in range(batch_number):
        train.run(feed_dict={x : input_shuffle[j*batch_size:(j+1)*batch_size]})
    print("Epoch({}/{}) cost : {} ".format(i,epoch, cost.eval(feed_dict={x : input_data})))

saver = tf.train.Saver()
saver.save(sess, os.path.join(save_dir, 'auto'))
