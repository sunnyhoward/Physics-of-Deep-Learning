import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, _) =  tf.keras.datasets.mnist.load_data()

train_filter = np.where((train_labels == 0))

train_data = train_data[train_filter]
train_data = train_data[0:1000]
plt.imshow(train_data[1])

#make the data binary. 
train_data[train_data<=127] = 0
train_data[train_data>127] = 1
plt.imshow(train_data[1])


new_data = train_data
         
            
for i in range(10):
    plt.figure()
    plt.imshow(new_data[i])

#flatten the data.
new_data = np.reshape(new_data,(new_data.shape[0],new_data.shape[1]*new_data.shape[2]))

############################
#Now Try to create the RBM

def calculatehidden(visible,weights):
    hidden = np.tensordot(visible,weights,((0),(0)))
    return hidden

def sigmoid(x):
    sigmoid = 1/(1+np.exp(x))
    return sigmoid 

def sampleprob(prob):
    numbers = np.random.uniform(0,1,len(prob))
    sample = np.asarray([prob>numbers]).astype(int)
    return sample
    
def entropyderivative(visible,weights):
    tanh = np.tanh(np.tensordot(weights,visible,((0),(0))))
    bmaverage = np.outer(visible,tanh)
    
def training(new_data,learningrate,weights,hiddenbias,visiblebias,batchsize,epochs):
    positivegrad = np.zeros((len(visiblebias),len(hiddenbias)))
    negativegrad = np.zeros((len(visiblebias),len(hiddenbias)))
    store1 = np.zeros(len(visiblebias))
    store2 = np.zeros(len(hiddenbias))
    sumweights = np.zeros(epochs)
    erroradd=np.zeros(epochs)
    
    for epoch in range(epochs):
        print('epoch ' + str(epoch))
        error = 0
        for i in range(len(new_data)):
            #put the image in the visible layer 
            v = new_data[i]
            
            #sample the hidden layer by first finding probabilities.
            probh = sigmoid(hiddenbias + np.tensordot(weights,v,((0),(0))))
            sampleh = sampleprob(probh).reshape((len(hiddenbias)))
            
            #now calculate positive gradient, add it to running total.
            positivegrad+= np.outer(v,sampleh)
            
            #now sample v from the h, followed by sampling h from v again.
            probv = sigmoid(visiblebias + np.tensordot(weights,sampleh,((1),(0))))
            samplev = sampleprob(probv).reshape((len(visiblebias)))
            probh2 = sigmoid(hiddenbias + np.tensordot(weights,samplev,((0),(0))))
            sampleh2 = sampleprob(probh2).reshape((len(hiddenbias)))
            
            #now calculate negative gradient, add to running total.
            negativegrad += np.outer(samplev,sampleh2)
            
            #store the values of the following in running total:
            store1+= v-samplev
            store2+= sampleh-sampleh2
            
            if i % batchsize == 0:
                #now update weights and biases.
                weights+= -learningrate * (positivegrad-negativegrad)/batchsize
                #reset positive and negative gradients.
                positivegrad = np.zeros((len(visiblebias),len(hiddenbias)))
                negativegrad = np.zeros((len(visiblebias),len(hiddenbias)))
                
                visiblebias+= -learningrate * (store1)/batchsize
                hiddenbias+= -learningrate * (store2)/batchsize
                
                store1 = np.zeros(len(visiblebias))
                store2 = np.zeros(len(hiddenbias))
                
            error+=np.sum((v-samplev)**2)
        print('at epoch ' + str(epoch) + ' error is ' + str(error))
        print(np.sum(weights))
        sumweights[epoch] = np.sum(weights)
        erroradd[epoch] = error
    return weights,visiblebias,hiddenbias,sumweights,erroradd


visiblebias = np.zeros(len(new_data[1]))

hiddenbias = np.zeros(int(len(new_data[1])*1.5))

weights = np.sqrt(0.01) * np.random.randn(len(visiblebias),len(hiddenbias))

batchsize = 100

learningrate = 0.01
epochs =30

weights,visiblebias,hiddenbias,sumweights,erroradd = training(new_data,learningrate,weights,hiddenbias,visiblebias,batchsize,epochs)

plt.figure()
plt.plot(range(epochs),sumweights)
plt.show()
plt.figure()
plt.plot(range(epochs),erroradd)
plt.show()

v = np.random.randint(2,size = (5,28**2))
for step in range(200):
  if step % 20 == 0:
    fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(20,4))  
    for i in range(5):
      axes[i].imshow(np.reshape(v[i],[28, 28]))
      axes[i].get_xaxis().set_visible(False)
      axes[i].get_yaxis().set_visible(False)
    plt.show()
  for i in range(5):
      probh = sigmoid(hiddenbias + np.tensordot(weights,v[i],((0),(0))))
      sampleh = sampleprob(probh).reshape((len(hiddenbias)))
      probv = sigmoid(visiblebias + np.tensordot(weights,sampleh,((1),(0))))
      v[i] = sampleprob(probv).reshape((len(visiblebias)))