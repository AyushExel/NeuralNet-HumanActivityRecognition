
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import csv
import numpy  as np
# Load a CSV file
def load_csv(filename):
	file = open(filename,  "rt")
	lines = reader(file)
	dataset = list(lines)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

#convert alphabetic labels to numeric for using as sigmoid function output
def alpha_to_num(labels,outputs):
	output_form = np.zeros((len(outputs),len(labels)))
	for i in range(len(outputs)):
		index = outputs[i]
		ind   = label[index[0]]
		output_form[i][ind] = 1
	return output_form
#initialize parameters for each layer
def init_theta(fin,fout):
	uniform_param = 4* np.sqrt(6/(fin+fout)) #parameter of uniform distribution to randomly intialize parameters
    #initialize theta array of size inputLayer*(1+outputLayer)
	theta = np.random.uniform(-uniform_param,uniform_param,(1+fin)*(fout))
	#here, adding 1 to output layer size ensures a bais term 
	theta = theta.reshape(1+fin,fout)
	return theta


eps = 0.01
def relu(arg):
    return np.maximum(eps*arg,arg)

def reluGrad(arg):
    ret = 1. *(arg>0)
    ret[ret==0] = eps
    return ret
 
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = x.transpose()
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum(axis=0)+ 1e-15).transpose()


#load data
data = load_csv("activity/train.csv")

#store data into numpy array
data = np.array(data[1:])

#randomly shuffle the data
np.random.shuffle(data)


#extract outputs in string format
outputString = data[:,[len(data[0])-1]]


#keep only the required numerical data in 'data' matrix
data = np.delete(data,[len(data[0])-1,len(data[0])-2],1)


#convert numerical data into correct data type
for i in range(len(data[0])-2):
	str_column_to_float(data,i)
data = data.astype(np.float64)
data = data.transpose()
data = (data-data.min(0))/data.ptp(0)
data = data.transpose() 
	
#create enumerations for different labels
label = {'WALKING':0,'WALKING_UPSTAIRS':1,'WALKING_DOWNSTAIRS':2,'SITTING':3,'STANDING':4,'LAYING':5}

#change labels to output arrays
output = alpha_to_num(label,outputString)

'''
SHAPES:
data => (7352, 561) 
output => (7352, 6)

NEURAL NETWORK DESCRIPTION:
input layer size => 561
1st hidden layer size =>10
output layer size => 6
'''

#initialize random thetas
theta1 = init_theta(561,90)
theta2 = init_theta(90,6)
iters = 800#works best with adagrad 96%
iters = 400
iters = 300
#lamb= 0.6
lamb =0.004

alpha = 0.03628# best 96% with adagrad and 800iters
alpha = 0.038221 
alpha = 0.01
#for momentum update
v1 = theta1*0
v2 = theta2*0
mu = 0.9

#for Adagrad update
cacheTheta1  = theta1*0
cacheTheta2  = theta2*0 

#for RMSprop update:
decay_rate = 0.8
v1 = theta1*0
v2 = theta2*0
#for Adam:
beta1 = 0.9
beta2 = 0.99
m1=0
m2 = 0
v1 = theta1*0
v2 = theta2*0
for i in range(1,iters):
    '''
    if i%15==0:
        a1 = np.insert(data,0,np.ones(len(data)),1).astype(np.float64)
        z2 = a1.dot(theta1).astype(np.float64)
        a2 = relu(z2).astype(np.float64)
        a2 = np.insert(a2,0,np.ones(len(a2)),1).astype(np.float64)
        z3 = a2.dot(theta2).astype(np.float64)
        a3 = softmax(z3).astype(np.float64)

        a3=(a3 == a3.max(axis=1)[:,None])
        count = 0.0
        for i in range(len(a3)-1):
	        if np.array_equal(a3[i],output[i]):
		        count = count+1
        print("Train accuracy => " , (count/len(a3))*100)
    '''
    a1 = np.insert(data,0,np.ones(len(data)),1).astype(np.float64)
    z2 = a1.dot(theta1).astype(np.float64)
    a2 = relu(z2)
    a2 = np.insert(a2,0,np.ones(len(a2)),1)
    z3 = a2.dot(theta2)
    a3 = softmax(z3)

    eps = 1e-8
    cost = -(output*(np.log(a3+eps))+(1-output)*(np.log(1-a3+eps))).sum()
    cost = (1/len(data))*cost + (lamb/(2*len(data)))*((np.delete(theta1,0,0)**2).sum() + (np.delete(theta2,0,0)**2).sum())
      
    sigma3 = a3-output
    sigma2 = (sigma3.dot(np.transpose(theta2)))* reluGrad(np.insert(z2,0,np.ones(len(z2)),1))
    sigma2 = np.delete(sigma2,0,1)
    delta2 = (np.transpose(a2)).dot(sigma3)
    delta1 = (np.transpose(a1)).dot(sigma2)

    grad1 = delta1/len(data) + (lamb/len(data))*np.insert(np.delete(theta1,0,0),0,np.zeros(len(theta1[0])),0)
    grad2 = delta2/len(data) + (lamb/len(data))*np.insert(np.delete(theta2,0,0),0,np.zeros(len(theta2[0])),0)
    #Adam update:
    m1 = beta1*m1 + (1-beta1)*grad1
    m2 = beta2*m2 + (1-beta1)*grad2
    mt1 = m1 / (1-beta1**i)
    mt2 = m2 / (1-beta1**i)
    v1 = beta2*v1 + (1-beta2)*(grad1**2)
    v2 = beta2*v2 + (1-beta2)*(grad2**2)
    vt1 = v1 / (1-beta2**i)
    vt2 = v2/(1-beta2**i)
    theta1 += - alpha * mt1 / (np.sqrt(vt1) + 1e-8)
    theta2 += - alpha * mt2 / (np.sqrt(vt2) + 1e-8) 


    #RMSprop update: BUGGY!!!!!!!1
    '''
    cacheTheta1 = decay_rate*cacheTheta1 + (1-decay_rate)*grad1**2
    cacheTheta2 = decay_rate*cacheTheta2 + (1-decay_rate)*grad2**2
    theta1 -= alpha*grad1/(np.sqrt(cacheTheta1 + 1e-7 ))
    theta2 -= alpha*grad2/(np.sqrt(cacheTheta2 + 1e-7 ))
    '''
    # momentum update : 97% cost=>0.14 currently works the best
    '''
    v1 =v1*mu - alpha*grad1
    v2 = v2*mu - alpha*grad2
    theta1 += v1
    theta2 += v2
    '''
    # Adagrad update: 94%
    '''
    cacheTheta1 += grad1**2
    cacheTheta2 += grad2**2
    theta1 -= alpha*grad1/(np.sqrt(cacheTheta1 + 1e-7 ))
    theta2 -= alpha*grad2/(np.sqrt(cacheTheta2 + 1e-7 ))
    '''
    #normal gradient descent: 92% cost=>0.39
    '''
    theta1 = theta1 - alpha*grad1
    theta2 = theta2 - alpha*grad2
    '''
    print("iteration ",i," cost => ",cost)

#accuracy of training set
a1 = np.insert(data,0,np.ones(len(data)),1).astype(np.float64)
z2 = a1.dot(theta1).astype(np.float64)
a2 = relu(z2).astype(np.float64)
a2 = np.insert(a2,0,np.ones(len(a2)),1).astype(np.float64)
z3 = a2.dot(theta2).astype(np.float64)
a3 = softmax(z3).astype(np.float64)

a3=(a3 == a3.max(axis=1)[:,None])
count = 0.0
for i in range(len(a3)-1):
	if np.array_equal(a3[i],output[i]):
		count = count+1
print("Train accuracy => " , (count/len(a3))*100)


data1 = load_csv("activity/test.csv")

#store data into numpy array
data1 = np.array(data1[1:])

#randomly shuffle the data


#extract outputs in string format
outputString = data1[:,[len(data1[0])-1]]


#keep only the required numerical data in 'data' matrix
data1 = np.delete(data1,[len(data1[0])-1,len(data1[0])-2],1)

#convert numerical data into correct data type
for i in range(len(data1[0])-2):
	str_column_to_float(data1,i)

data1 = data1.transpose().astype(np.float64)
data1 = (data1-data1.min(0))/data1.ptp(0)
data1 = data1.transpose()
outputtest = alpha_to_num(label,outputString)

a1 = np.insert(data1,0,np.ones(len(data1)),1).astype(np.float64)
z2 = a1.dot(theta1).astype(np.float64)
a2 = relu(z2).astype(np.float64)
a2 = np.insert(a2,0,np.ones(len(a2)),1).astype(np.float64)
z3 = a2.dot(theta2).astype(np.float64)
a3 = softmax(z3).astype(np.float64)

a3=(a3 == a3.max(axis=1)[:,None])
count = 0.0
for i in range(len(a3)-1):
	if np.array_equal(a3[i],outputtest[i]):
		count = count+1
print("Test accuracy => " , (count/len(a3))*100)
