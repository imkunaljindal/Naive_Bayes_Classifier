#!/usr/bin/env python
# coding: utf-8
# @author: Kunal Jindal

# # Naive Bayes - Mushroom Dataset
# - Goal is to predict class of mushrooms, given some features of mushrooms.</br> 'Naive Bayes' Model for this classification has been used to achieve the goal.

# ### Load the Dataset



import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('Mushrooms/mushrooms.csv')



#df.head(10)
#print(df.shape)



# ## Encode the Categorical Data into Numerical Data



le = LabelEncoder()

#Applies transformation on each column
ds = df.apply(le.fit_transform)


'''
data = ds.values
print(data.shape)
print(type(data))
print(data[:5,:])

data_y = data[:,0]
data_x = data[:,1:]

'''


# ## Break the Data into Train and Test 


x_train,x_test , y_train,y_test = train_test_split(data_x,data_y,test_size=0.2)




#print(x_train.shape,y_train.shape)
#print(x_test.shape,y_test.shape)



#np.unique(y_train)




# # Building Our Classifier!


def prior_prob(y_train,label):
    
    total_examples = y_train.shape[0]
    class_examples = np.sum(y_train==label)
    
    return (class_examples)/float(total_examples)




def cond_prob(x_train,y_train,feature_col,feature_val,label):
    
    x_filtered = x_train[y_train==label]
    numerator = np.sum(x_filtered[:,feature_col]==feature_val)
    denominator = np.sum(y_train==label)
    
    return numerator/float(denominator)




# ## Next Step: Compute Posterior Prob for each test example and make predictions



def predict(x_train,y_train,xtest):
    """ XTest is a single testing point """
    classes = np.unique(y_train)
    n_features = x_train.shape[1]
    post_probs = [] #List of probs for all classes and given a single testing point
    #Compute Posterior for each class 
    for label in classes:
        
        #Postc = likelihood*prior
        likelihood = 1.0
        for f in range(n_features):
            cond = cond_prob(x_train, y_train,f,xtest[f],label)
            likelihood *= cond
            
        prior = prior_prob(y_train,label)
        post = likelihood*prior
        post_probs.append(post)
    
    pred = np.argmax(post_probs)
    return pred



output = predict(x_train,y_train,x_test[1])
print(output)
print(y_test[1])


# ## Calculating the Accuracy of our Classifier



def score(x_train,y_train,x_test,y_test):
    
    pred = []
    
    for i in range(x_test.shape[0]):
        pred_label = predict(x_train,y_train,x_test[i])
        pred.append(pred_label)
        
    pred = np.array(pred)
    
    accuracy = np.sum(pred==y_test)/y_test.shape[0]
    return accuracy


print(score(x_train,y_train,x_test,y_test))





