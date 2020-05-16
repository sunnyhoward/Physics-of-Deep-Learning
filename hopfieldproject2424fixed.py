import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def createpatterns():
    (train_data, train_labels), (test_data, _) =  tf.keras.datasets.mnist.load_data()
    
    train_data = train_data[0:5000]
    train_labels = train_labels[0:5000]
    
    #make the data binary. 
    train_data[train_data<=127] = 1
    train_data[train_data>127] = 0
    
    new_data = train_data
    
    zero_filter = np.where(train_labels == 0)
    one_filter = np.where(train_labels == 1)
    two_filter = np.where(train_labels == 2)
    three_filter = np.where(train_labels == 3)
    four_filter = np.where(train_labels == 4)
    five_filter = np.where(train_labels == 5)
    six_filter = np.where(train_labels == 6)
    seven_filter = np.where(train_labels == 7)
    eight_filter = np.where(train_labels == 8)
    nine_filter = np.where(train_labels == 9)
    
    zero_data = new_data[zero_filter]
    one_data = new_data[one_filter]
    two_data = new_data[two_filter]
    three_data = new_data[three_filter]
    four_data = new_data[four_filter]
    five_data = new_data[five_filter]
    six_data = new_data[six_filter]
    seven_data = new_data[seven_filter]
    eight_data = new_data[eight_filter]
    nine_data = new_data[nine_filter]
    
    #choosing the patterns
    pattern0 = zero_data[4].astype(float)
    pattern1 = one_data[0].astype(float)
    pattern2 = two_data[0].astype(float)
    pattern3 = three_data[0].astype(float)
    pattern4 = four_data[0].astype(float)
    pattern5 = five_data[0].astype(float)
    pattern6 = six_data[0].astype(float)
    pattern7 = seven_data[0].astype(float)
    pattern8 = eight_data[0].astype(float)
    pattern9 = nine_data[0].astype(float)

    pattern0[pattern0==0] = -1
    pattern1[pattern1==0] = -1
    pattern2[pattern2==0] = -1
    pattern3[pattern3==0] = -1
    pattern4[pattern4==0] = -1
    pattern5[pattern5==0] = -1
    pattern6[pattern6==0] = -1
    pattern7[pattern7==0] = -1
    pattern8[pattern8==0] = -1
    pattern9[pattern9==0] = -1

    
    pattern0 = np.reshape(pattern0,(new_data.shape[1]*new_data.shape[2]))
    pattern1 = np.reshape(pattern1,(new_data.shape[1]*new_data.shape[2]))
    pattern2 = np.reshape(pattern2,(new_data.shape[1]*new_data.shape[2]))
    pattern3 = np.reshape(pattern3,(new_data.shape[1]*new_data.shape[2]))
    pattern4 = np.reshape(pattern4,(new_data.shape[1]*new_data.shape[2]))
    pattern5 = np.reshape(pattern5,(new_data.shape[1]*new_data.shape[2]))
    pattern6 = np.reshape(pattern6,(new_data.shape[1]*new_data.shape[2]))
    pattern7 = np.reshape(pattern7,(new_data.shape[1]*new_data.shape[2]))
    pattern8 = np.reshape(pattern8,(new_data.shape[1]*new_data.shape[2]))
    pattern9 = np.reshape(pattern9,(new_data.shape[1]*new_data.shape[2]))

    return pattern0,pattern1,pattern2,pattern3,pattern4,pattern5,pattern6,pattern7,pattern8,pattern9

def createconnectivity(invQ,pattern,pattern1):
    print('creating connectivity matrix')
    J = np.zeros((len(pattern1),len(pattern1)))
    for x in range(len(pattern1)):
        print(str(int(x/len(pattern1) * 100)) + '% done')
        for y in range(len(pattern1)):
            for a in range(len(pattern)):
                for b in range(len(pattern)):
                    J[x,y] += pattern[a][x]*invQ[a,b]*pattern[b][y]
    J = J/len(pattern1)
    return J
    
def calculateenergy(state,J):
    energy = 0
    for i in range(len(state)):
        for j in range(len(state)):
            if i !=j:
                energy+=J[i,j] * state[i] * state[j]
    return -1/2 * energy

def calculateenergychange(state,spin,J):
    origenergy = 0
    newenergy = 0
    for i in range(len(state)):
        if i != spin:
            origenergy+=-J[spin,i]*state[spin]*state[i]
            newenergy+=-J[spin,i]*-state[spin]*state[i]
    energychange = newenergy - origenergy
    return energychange
    

def update(pattern1,state,J,b):
    #first choose random cell
    spin = np.random.randint(0,len(pattern1))
    energychange = calculateenergychange(state, spin, J)
    if energychange <= 0:
        state[spin] = -state[spin]
    elif np.random.uniform()<np.exp(-b*energychange) :
        state[spin] = -state[spin]
    return state

def classifypattern(pattern,state): 
    overlap = np.absolute(np.tensordot(pattern,state,((1),(0))))
    assign = np.argmax(overlap)
    return assign

pattern0,pattern1,pattern2,pattern3,pattern4,pattern5,pattern6,pattern7,pattern8,pattern9 = createpatterns()
#############################################
pattern = [pattern0,pattern1,pattern2,pattern3,pattern4,pattern5,pattern6,pattern7,pattern8,pattern9]

Q = np.zeros((10,10))
for i in range(10):
    for j in range(10):
        Q[i,j] = np.tensordot(pattern[i],pattern[j],((0),(0)))/len(pattern1) #- np.sum(pattern[i])*np.sum(pattern[j]) /(len(pattern1)**2)
        
invQ = np.linalg.inv(Q)

# fig, ax = plt.subplots()
# ax.set_title('Showing Connected Correlation between Patterns')
# ax.set_xticks(np.arange(0, 10))
# ax.set_yticks(np.arange(0, 10))
# ax.set_xlabel('Pattern')
# ax.set_ylabel('Pattern')
# ax.imshow(overlapgrid)

# for i in range(10):
#     for j in range(10):
#         if i>=j:
#             ax.text(i,j,str(np.round(overlapgrid[i,j],2)), ha="center", va="center", color="w",fontsize=8)
# plt.show()



J = createconnectivity(invQ,pattern,pattern1)

#initialize state:
state = np.random.randint(0,2,size = len(pattern1))

# state = v[2]
state[state==0] = -1

plt.imshow(np.reshape(state,[28, 28]))
plt.title('Initial State')
plt.show()

energy = calculateenergy(state,J)

b = 10
t=0
T=8
n = 0
while t <T:
    state = update(pattern1,state,J,b)
    t += 1/len(pattern1)
    n+=1
    if n%100 == 0:
        plt.imshow(np.reshape(state,[28, 28]))
        plt.title('Intermediate State')
        plt.show()

classify  = classifypattern(pattern, state)
print('Classify as ' + str(classify))
# for i in range(10):
#     print('overlap with pattern'+ str(i) +' = ' + str(np.tensordot(pattern[i],state,((0),(0)))/len(pattern1)))