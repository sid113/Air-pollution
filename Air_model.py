import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#importing libraries and data
dataset=pd.read_csv('Train.csv');
#print(dataset.head());
#print(dataset.shape);
#print(dataset.describe());
#print(dataset.isnull().any());


X = dataset[['feature_1', 'feature_2', 'feature_3', 'feature_4','feature_5']].values
Y = dataset['target'].values

#dividing our dataset into train dataset and test dataset as per 80 20 rule
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#training phase 
regressor=LinearRegression()
regressor.fit(X,Y);
columns_dataset=np.array(['feature_1', 'feature_2', 'feature_3', 'feature_4','feature_5']);
weights=pd.DataFrame(regressor.coef_, index=columns_dataset, columns=['Coefficient'])
print(weights);

#testing phase

print("hwjehwjejwejhwjh");
y_hat=regressor.predict(X_test);
loss=pd.DataFrame({'Acutal':y_test,'predicted':y_hat});
print(loss);
top_five=loss.head();
top_five.plot(kind='bar',figsize=(10,8))
plt.show();


