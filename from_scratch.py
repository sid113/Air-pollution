import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle

#importing libraries and data
dataset=pd.read_csv('Train.csv');
#print(dataset.head());
#print(dataset.shape);
#print(dataset.describe());
#print(dataset.isnull().any());


X = dataset[['feature_1', 'feature_2', 'feature_3', 'feature_4','feature_5']].values
Y = dataset['target'].values

X = (X - X.mean())/X.std()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train, y_train = shuffle(X_train, y_train)

print(y_train.shape)

y_train=y_train.reshape((1280, 1))
Y=Y.reshape((1600, 1))
#Normalization


#initilizing hyper parameters
learning_rate=0.001;
epochs=10000;

#initilizing weights
theta=np.zeros([1,5]);
bias=0;


#cost function
def MSE(X_train,y_train,theta):
    #theta = theta - (learning_rate/len(X_train)) * np.sum( X_train((X_train @ theta.T)- y_train), axis=0)  
    y_prediction=X_train @ theta.T+bias  
    square=np.power((y_prediction-y_train),2);
    return np.sum(square)/(2*len(X_train));
    
    
    
# gradient descent the optimizer algorithm to minimize the cost function
def gradientDescent(X,y_train,theta,bias):
    cost=np.zeros(epochs);
    for i in range(0,epochs):
        #print(add)
        y_prediction=X @ theta.T+bias
        theta = theta - (learning_rate/len(X)) * np.sum(X * (y_prediction - y_train), axis=0)
        bias= bias- (learning_rate/len(X)) * np.sum(y_prediction-y_train)
       
        cost[i]=MSE(X,y_train,theta);
        print("Epoch ",i+1);
    return theta,cost,bias

weights,cost,bias=gradientDescent(X,Y,theta,bias)


test_dataset=pd.read_csv('Test.csv');
#print(dataset.head());
#print(dataset.shape);
#print(dataset.describe());
#print(dataset.isnull().any());


X_test =test_dataset[['feature_1', 'feature_2', 'feature_3', 'feature_4','feature_5']].values

y_hat=X_test @ weights.T+bias;
print(y_hat.shape);

y_hat=y_hat.reshape((400,));
ID=np.arange(1, 401)
loss=pd.DataFrame({ 'ID':ID,'target':y_hat});
print(loss);

plt.style.use('seaborn')
plt.plot(np.arange(epochs),cost,'r');
plt.xlabel('Epochs')
plt.ylabel('Cost');
plt.title('Error vs. Training Epoch');
#plt.show();

export_csv = loss.to_csv (r'ans.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

#print (df)


