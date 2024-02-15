# Decision-Trees-Algo

The splitting criteria at the root could be the same or different.
the split at the root node of a decision tree might change depending on the characteristics of the dataset being used. The root node is the first node in a decision tree and is used to make the first decision about which direction to follow in the tree based on the features of the data. The root node is typically split into two or more child nodes based on a chosen feature and a split point. The decision about which feature and split point to use is made using a measure of feature importance and a criterion for evaluating the splits, such as maximizing the purity of the resulting nodes.
In general, the split at the root node of a decision tree is likely to change if the dataset has a small number of points or if the features of the data are highly correlated with the target variable. This is because a small number of points may not provide enough information to make an informed decision about the best split, and highly correlated features may dominate the decision-making process. On the other hand, if the dataset has a large number of points and the features are less correlated with the target variable, the split at the root node is likely to be more stable.

The true statements regarding overfitting in a decision tree are:
Over-attempting to get more and more pure nodes leads to overfitting.
Explanation: When a decision tree tries to create very pure nodes by splitting the data too finely, it can end up fitting the noise in the training data, leading to overfitting.
Low values of the 'min_samples_split' parameter in the Decision Tree class of sklearn lead to overfitting.
Explanation: A low value of 'min_samples_split' allows the tree to split nodes even for a small number of data points, which can lead to overfitting by creating very deep and complex trees.

Over-attempting to get more and more pure nodes leads to overfitting.
Low values of the ‘ min_samples_split ’ parameter in the Decision Tree class of sklearn lead to overfitting.
Explanation :
A node is 100 % pure when all of its data belongs to a single class. If a decision tree model tries to attempt more and more pure nodes, it means that instead of generalizing it is trying to be highly accurate on the training data. This is nothing but overfitting the data.
The parameter 'min_samples_split' is used to set the minimum number of data points required to split further. This means that the tree will stop splitting if the number of observations left is not greater than min_samples_split. This helps to control depth which therefore prevents overfitting. Thus, if the value of this parameter is set low, the tree will keep splitting more and more which will lead to overfitting.
The 'max_depth' parameter represents how deep your decision tree will be. The more you decrease this number, the less the number of splits, which will lead to a less complex and more generalized model. Thus, Low values of this parameter do not lead to overfitting.
A decision tree model should have optimized maximum depth and a minimum number of points to split a node. With this, high values of depth will lead to overfitting, and low values of depth will lead to underfitting.

#Entropy : If there are only positive examples or only negative examples, then Entropy=0.
If there are an equal number of positive and negative examples, Then Entropy=1.
Explanation :
For a dataset where all the observations belong to a single class, Entropy =−((1).log2(1) = 0. Such a dataset has no impurity.
However, if we have a dataset with two classes, half made up of yellow and the other half being purple, the entropy will be one. Entropy = −(((0.5).log2(0.5)+((0.5).log2(0.5)) = 1
Calculate Entropy: 
Code-------import numpy as np
---------------------------Given the list of "class" labels, complete the function to return the entropy rounded up to two decimal places.               
def entropy(y_target):
    '''
    Calculates the entropy given list of target(binary) variables
    '''
    # Write your code here
    
    # Initialize the entropy
    entropy = 0
    
    # Calculate the counts of each unique element in y_target
    unique_elements, counts = np.unique(y_target, return_counts=True)
    
    # Probabilities of each class label
    prob = counts / len(y_target)
    
    # Calculate the entropy involving all the unique elements
    for p in prob:
        entropy -= p * np.log2(p) if p > 0 else 0
    
    return np.round(entropy, 2)

    Problem Description:

Given the class_vector in the form of a list consisting of the target variables for the observations, complete the function to return the ##gini_impurity of the class_vector rounded up to two decimal places.
Note: The target variable is a binary variable (0/1)

from collections import Counter
import numpy as np

def gini_impurity(class_vector):  
    # Gini impurity = 1−∑pi2
    #dictionary with unique values and counts
    counts = Counter(class_vector)      
    
    #probability of class 0 
    prob_zero = counts.get(0, 0) / len(class_vector) 
    
    #probability of class 1
    prob_one = counts.get(1, 0) / len(class_vector)    
    
    #probability square sum
    prob_sqrsum = prob_zero**2 + prob_one**2 
    
     #Calculate the gini impurity
    gini_imp = 1 - prob_sqrsum   
    
    return np.round(gini_imp,2)

Q.. In a set of data points, the entropy would be
    lower in the case of a dominant class
    higher in the case of equiprobable classes
    Explanation :

        The formula for entropy is: E=∑−pi.log2pi.
 If we calculate the entropy for an equiprobable two classes distribution we will get the entropy as 1 i.e. −21.lo2(21)−21.log2(21) = 1.If we increase the percentage of first-class by 20%, then the new entropy 
 will be −(107).log2(107)−103.log2(103) = −(−0.51).(0.7) -(−1.74).(0.3) = 0.879When a class is 90% of the data, we will get the entropy as −(0.9).log2(0.9)−(0.1).log2(0.1) = −(−0.9).(0.15)−(0.1).(−3.32) = 0.467
 Thus, as a class starts becoming dominant the entropy decreases.

 Q. What can be said about the Gini Impurity of a child node when compared to its parent node?
    Note: Here we are considering the Gini impurity of an individual child node, not the weighted split impurity. For this question, consider the case of A, B, A, A, A with this order only, and now try to 
    answer the question.
                             : It may have a higher or lower Gini impurity

                             Explanation:

A node’s Gini impurity is generally lower than that of its parent as the CART training algorithm cost function splits each of the nodes in a way that minimizes the weighted sum of its children’s Gini impurities. However, sometimes it is also possible for a node to have a higher Gini impurity than its parent but in such cases, the increase is more than compensated by a decrease in the other child’s impurity.
For better understanding let’s consider the following Example:

Consider a node containing four samples of class A and one sample of class B.

Then, its Gini impurity is calculated as 1–(51)2–(54)2=0.32

Now suppose the dataset is one-dimensional and the instances are arranged in the manner: A, B, A, A, A.

We can verify that the algorithm will split this node after the second instance, producing one child node with instances A, B,
and another child node with instances A, A, A.
Then, the first child node’s Gini impurity is 1–(21)2–(21)2=0.5, which is higher than its parent’s.
This is compensated for by the other node being pure, so its overall weighted Gini impurity is  :52×0.5+ 53×0=0.2, which is lower than the parent’s Gini impurity.

When 50% of observations belong to y+, then, the Gini impurity of the system would be 0.5
When 0% of observations belong to y- , then, the Gini impurity of the system would be 0


The splitting criteria at the root could be the same or different.
Explanation

The split at root of decision tree might or might not change depending on your dataset. The split at root of decision tree is likely to change if your dataset has a small number of points.
For example, let T be training set with one continuous attribute A and a binary target class C. Let us use the Gini gain Δ as a splitting criterion - see an example here. Let’s say you have two duplicated points x1 and x2 with x1= x2
If your dataset is:

A C
1 +
1 +
2 -

Then there will be no difference by removing one duplicated point (e.g. the first one). The dataset will be split into two sets according to the cut-off 1 for A anyway.

If your dataset is:

A C
1 +
1 +
2 -
3 +
4 +
5 -

The split induced on this dataset will be different from the split induced on the dataset where we remove the duplicated point (e.g. the first one).
The split induced on this dataset will be found by optimizing Δ on all possible cut-offs for A. If you do it, you will see that the best Δ is obtained with 1 as the cut-off for A. (Δ=0.11);
If we remove the first point, the optimization procedure will select the cut-off 4 for A. The cut-off 1 has Δ=0.08, the cut-off on 2 has Δ=0.013, and the cut-off 4 has Δ=0.18.
The two examples above are both related to a small number of points. You can imagine that what happened in the second example is more likely to happen if the number of data points is small.

Q. According to the value of ____, we split the node and split the decision tree?
Information-gain
Explanation
We use the Gini-Impurity and Entropy to calculate the information gain when using a feature for splitting
Hence Information gain is the one which decides which feature is best for splitting the tree

Q. Decision Tree Classifier :

      import numpy as np

observation = eval(input())

"""
'observation' IS THE OBSERVATION THAT YOU HAVE TO PREDICT.
DO NOT CHANGE THE ABOVE CODE
An example of 'observation' variable: [[5,15,20]]
"""

X_train = np.asarray(X)
y_train = np.asarray(y)


#IMPORT DECISION TREE CLASSIFICATION MODEL
from sklearn.tree import  DecisionTreeClassifier

#INITIALIZE DECISION TREE CLASSIFICATION MODEL
classifier = DecisionTreeClassifier()
#TRAIN DECISION TREE CLASSIFICATION MODEL
classifier.fit(X_train, y_train)
#PRINT PREDICTED VALUE BY THE MODEL FOR the variable 'observation'
predicted_value = classifier.predict(observation)
print(predicted_value)

a - Overfitting; b- Underfitting
Explanation:

We can set maximium depth upto which a tree can split by using ‘max_depth’.

‘ min_samples_split ‘ is used to set the minimum number of data points required to split further. which helps us to control depth which therefore prevents overfit.

A decision tree model should have optimized maximum depth and minimum number of points to split a node. With this, option ‘a’ will lead to overfitting, option ‘b’ will be underfitting the data.







    

