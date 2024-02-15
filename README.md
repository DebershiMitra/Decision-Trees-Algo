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
