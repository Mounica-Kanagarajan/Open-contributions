# **MACHINE LEARNING**

In machine learning, we use past data to predict a future state. When data is labelled based on a desired attribute, 
we call it supervised learning. There are many algorithms facilitating such a learning. Decision tree is one such. 
Decision tree is a directed graph where nodes correspond to some test on attributes, branch represents an outcome of a test and a leaf
corresponds to a class label.

## **DECISION TREE**

Decision tree learning is one of the most widely used and practical methods for inductive inference.It is a method for approximating 
discrete-valued target functions, in whichthe learned function is represented by a decision tree. Learned trees can also be 
re-represented as sets of if-then rules to improve human readability.These learning methods are among the most popular of 
inductive inference algorithms.

The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by 
learning simple decision rules inferred from prior data(training data).In Decision Trees, for predicting a class label for a record
we start from the root of the tree. We compare the values of the root attribute with the record’s attribute. 
On the basis of comparison, we follow the branch corresponding to that value and jump to the next node.

## **Types of Decision Trees**
Types of decision trees are based on the type of target variable we have. It can be of two types:

**Categorical Variable Decision Tree** 
Decision Tree which has a categorical target variable then it called a Categorical variable decision tree.

**Continuous Variable Decision Tree**
Decision Tree has a continuous target variable then it is called Continuous Variable Decision Tree

## **Decision Tree Representation**
Decision trees classify instances by sorting them down the tree from the root to some leaf node, which provides the classification of 
the instance. Each node in thetree specifies a test of some attribute of the instance, and each branch descending from that node 
corresponds to one of the possible values for this attribute. Aninstance is classified by starting at the root node of the tree, testing
the attribute specified by this node, then moving down the tree branch corresponding to thevalue of the attribute in the given example.
This process is then repeated for the subtree rooted at the new node.

![image](https://devopedia.org/images/article/168/2041.1555312516.png)

## **Important Terminology**
- **Root Node:** It represents the entire population or sample and this further gets divided into two or more homogeneous sets.
- **Splitting:** It is a process of dividing a node into two or more sub-nodes.
- **Decision Node:** When a sub-node splits into further sub-nodes, then it is called the decision node.
- **Leaf / Terminal Node:** Nodes do not split is called Leaf or Terminal node.
- **Pruning:** When we remove sub-nodes of a decision node, this process is called pruning. You can say the opposite process of splitting.
- **Branch / Sub-Tree:** A subsection of the entire tree is called branch or sub-tree.
- **Parent and Child Node:** A node, which is divided into sub-nodes is called a parent node of sub-nodes whereas sub-nodes are the child of a parent node.

## **ID3 Algorithm**
 Decision tree algorithms transfom raw data to rule based decision making trees. Herein, ID3 is one of the most common decision tree algorithm. 
 Firstly, It was introduced in 1986 and it is acronym of Iterative Dichotomiser.

## **Metrics in ID3**

- **Entropy** is the measure of disorder and the Entropy of a dataset is the measure of disorder in the target feature of the dataset.
  In the case of binary classification (where the target column has only two types of classes) entropy is 0 if all values in the target column are 
  homogenous(similar) and will be 1 if the target column has equal number values for both the classes.
            
           Entropy is calculated as:
                                     Entropy(S) = - ∑ pᵢ * log₂(pᵢ) ; i = 1 to n
                
                where,
                      -n is the total number of classes in the target column (in our case n = 2 i.e YES and NO)
                      -pᵢ is the probability of class ‘i’ or the ratio of “number of rows with class i in the target column” to 
                      the “total number of rows” in the dataset.

- **Information Gain** calculates the reduction in the entropy and measures how well a given feature separates or classifies the target classes. 
   The feature with the highest Information Gain is selected as the best one.
            
            Information Gain for a feature column A is calculated as:
                                      IG(S, A) = Entropy(S) - ∑((|Sᵥ| / |S|) * Entropy(Sᵥ))
                  
                  where,
                      -Sᵥ is the set of rows in S for which the feature column A has value v, 
                      |Sᵥ| is the number of rows in Sᵥ and likewise |S| is the number of rows in S
 
## **ID3 Steps:**
1. Calculate the Information Gain of each feature.
2. Considering that all rows don’t belong to the same class, split the dataset S into subsets using the feature for which the Information Gain is maximum.
3. Make a decision tree node using the feature with the maximum Information gain.
4. If all rows belong to the same class, make the current node as a leaf node with the class as its label.
5. Repeat for the remaining features until we run out of all features, or the decision tree has all leaf nodes.

## **Characteristics of ID3 Algorithm**
- ID3 uses a greedy approach that's why it does not guarantee an optimal solution; it can get stuck in local optimums.
- ID3 can overfit to the training data (to avoid overfitting, smaller decision trees should be preferred over larger ones).
- This algorithm usually produces small trees, but it does not always produce the smallest possible tree.
- ID3 is harder to use on continuous data (if the values of any given attribute is continuous, then there are many more places to split the data
  on this attribute, and searching for the best value to split by can be time consuming).


## **Overfitting**
- It is the transformation of learning into memorization.
- The performance graph is that after a certain moment, the increase in performance stops and starts to decrease.
- When your success rate high for training performance but opposite for test performance is an example for overfitting.
- Moreover, if there is a noisy sample in our data, it may cause overfitting.
- When the tree grows too large, a result is produced for almost every possible branch, which leads to overfitting.
- In order to avoid overfitting, it is necessary to prun the tree either when it is growing or after the tree is formed.

## **ID3 Algorithm Data**

**Using ID3 Algorithm to build a Decision Tree to predict the weather**

![image](https://iq.opengenus.org/content/images/2019/06/dataset.png)

Here,dataset is of binary classes(yes and no), where 9 out of 14 are "yes" and 5 out of 14 are "no".

Complete entropy of dataset is -
             H(S) = - p(yes) * log2(p(yes)) - p(no) * log2(p(no))
                  = - (9/14) * log2(9/14) - (5/14) * log2(5/14)
                  = - (-0.41) - (-0.53)
                  = 0.94
                  
For each attribute of the dataset, let's follow the step-2 of pseudocode : -

- **First Attribute - Outlook**

  **Categorical values - sunny, overcast and rain**

             H(Outlook=sunny) = -(2/5)*log(2/5)-(3/5)*log(3/5) =0.971
             H(Outlook=rain) = -(3/5)*log(3/5)-(2/5)*log(2/5) =0.971
             H(Outlook=overcast) = -(4/4)*log(4/4)-0 = 0

      Average Entropy Information for Outlook - 
      
            I(Outlook) = p(sunny) * H(Outlook=sunny) + p(rain) * H(Outlook=rain) + p(overcast) * H(Outlook=overcast)
                       = (5/14)*0.971 + (5/14)*0.971 + (4/14)*0
                       = 0.693

      Information Gain = H(S) - I(Outlook)
                       = 0.94 - 0.693
                       = 0.247


- **Second Attribute - Temperature**
 
  **Categorical values - hot, mild, cool**

              H(Temperature=hot) = -(2/4)*log(2/4)-(2/4)*log(2/4) = 1
              H(Temperature=cool) = -(3/4)*log(3/4)-(1/4)*log(1/4) = 0.811
              H(Temperature=mild) = -(4/6)*log(4/6)-(2/6)*log(2/6) = 0.9179
               
       Average Entropy Information for Temperature - 
  
            I(Temperature) = p(hot)*H(Temperature=hot) + p(mild)*H(Temperature=mild) + p(cool)*H(Temperature=cool)
                           = (4/14)*1 + (6/14)*0.9179 + (4/14)*0.811
                           = 0.9108

       Information Gain = H(S) - I(Temperature)
                      = 0.94 - 0.9108
                      = 0.0292
                      
- **Third Attribute - Humidity**

  **Categorical values - high, normal**
    
                 H(Humidity=high) = -(3/7)*log(3/7)-(4/7)*log(4/7) = 0.983
                 H(Humidity=normal) = -(6/7)*log(6/7)-(1/7)*log(1/7) = 0.591

          Average Entropy Information for Humidity - 

          I(Humidity) = p(high)*H(Humidity=high) + p(normal)*H(Humidity=normal)
                      = (7/14)*0.983 + (7/14)*0.591 
                      = 0.787

         Information Gain = H(S) - I(Humidity)
                          = 0.94 - 0.787
                          = 0.153
                        
- **Fourth Attribute - Wind**

  **Categorical values - weak, strong**

                  H(Wind=weak) = -(6/8)*log(6/8)-(2/8)*log(2/8) = 0.811
                  H(Wind=strong) = -(3/6)*log(3/6)-(3/6)*log(3/6) = 1

       Average Entropy Information for Wind - 
     
                I(Wind) = p(weak)*H(Wind=weak) + p(strong)*H(Wind=strong)
                        = (8/14)*0.811 + (6/14)*1 
                        = 0.892

        Information Gain = H(S) - I(Wind)
                         = 0.94 - 0.892
                         = 0.048
Here, the attribute with maximum information gain is Outlook. So, the decision tree built so far -
 
 ![image](https://iq.opengenus.org/content/images/2019/06/Untitled-Diagram--2-.png)

Here, when Outlook == overcast, it is of pure class(Yes).
Now, we have to repeat same procedure for the data with rows consist of Outlook value as Sunny and then for Outlook value as Rain.



 
Now, finding the best attribute for splitting the data with Outlook=Sunny values{ Dataset rows = [1, 2, 8, 9, 11]}.

  Complete entropy of Sunny is -
  
                  H(S) = - p(yes) * log2(p(yes)) - p(no) * log2(p(no))
                       = - (2/5) * log2(2/5) - (3/5) * log2(3/5)
                       = 0.971
     
- **First Attribute - Temperature**

  **Categorical values - hot, mild, cool**

                     H(Sunny, Temperature=hot) = -0-(2/2)*log(2/2) = 0
                     H(Sunny, Temperature=cool) = -(1)*log(1)- 0 = 0
                     H(Sunny, Temperature=mild) = -(1/2)*log(1/2)-(1/2)*log(1/2) = 1
                     
       Average Entropy Information for Temperature - 
       
          I(Sunny, Temperature) = p(Sunny, hot)*H(Sunny, Temperature=hot) + p(Sunny, mild)*H(Sunny, Temperature=mild) + p(Sunny, cool)*H(Sunny, Temperature=cool)
                                = (2/5)*0 + (1/5)*0 + (2/5)*1
                                = 0.4

        Information Gain = H(Sunny) - I(Sunny, Temperature)
                            = 0.971 - 0.4
                            = 0.571
                      
                 
- **Second Attribute - Humidity**

  **Categorical values - high, normal**

                    H(Sunny, Humidity=high) = - 0 - (3/3)*log(3/3) = 0
                    H(Sunny, Humidity=normal) = -(2/2)*log(2/2)-0 = 0

        Average Entropy Information for Humidity -
      
          I(Sunny, Humidity) = p(Sunny, high)*H(Sunny, Humidity=high) + p(Sunny, normal)*H(Sunny, Humidity=normal)
                             = (3/5)*0 + (2/5)*0 
                             = 0

        Information Gain = H(Sunny) - I(Sunny, Humidity)
                       = 0.971 - 0
                       = 0.971
                       
- **Third Attribute - Wind**

  **Categorical values - weak, strong**

                    H(Sunny, Wind=weak) = -(1/3)*log(1/3)-(2/3)*log(2/3) = 0.918
                    H(Sunny, Wind=strong) = -(1/2)*log(1/2)-(1/2)*log(1/2) = 1

        Average Entropy Information for Wind -
       
              I(Sunny, Wind) = p(Sunny, weak)*H(Sunny, Wind=weak) + p(Sunny, strong)*H(Sunny, Wind=strong)
                             = (3/5)*0.918 + (2/5)*1 
                             = 0.9508

        Information Gain = H(Sunny) - I(Sunny, Wind)
                        = 0.971 - 0.9508
                        = 0.0202
                 
Here, the attribute with maximum information gain is Humidity. So, the decision tree built so far -

![image](https://iq.opengenus.org/content/images/2019/06/Untitled-Diagram--3-.png)

Here, when Outlook = Sunny and Humidity = High, it is a pure class of category "no".
And When Outlook = Sunny and Humidity = Normal, it is again a pure class of category "yes". 
Therefore, we don't need to do further calculations.

Now, finding the best attribute for splitting the data with Outlook=Sunny values{ Dataset rows = [4, 5, 6, 10, 14]}.

                    Complete entropy of Rain is -
                    
                    H(S) = - p(yes) * log2(p(yes)) - p(no) * log2(p(no))
                         = - (3/5) * log(3/5) - (2/5) * log(2/5) 
                         = 0.971

- **First Attribute - Temperature**

  **Categorical values - mild, cool**

                    H(Rain, Temperature=cool) = -(1/2)*log(1/2)- (1/2)*log(1/2) = 1
                    H(Rain, Temperature=mild) = -(2/3)*log(2/3)-(1/3)*log(1/3) = 0.918

       Average Entropy Information for Temperature - 
      
                    I(Rain, Temperature) = p(Rain, mild)*H(Rain, Temperature=mild) + p(Rain, cool)*H(Rain, Temperature=cool)
                                         = (2/5)*1 + (3/5)*0.918
                                         = 0.9508

       Information Gain = H(Rain) - I(Rain, Temperature)
                        = 0.971 - 0.9508
                        = 0.0202
                        
- **Second Attribute - Wind**

  **Categorical values - weak, strong**
  
                    H(Wind=weak) = -(3/3)*log(3/3)-0 = 0
                    H(Wind=strong) = 0-(2/2)*log(2/2) = 0

         Average Entropy Information for Wind - 
     
                     I(Wind) = p(Rain, weak)*H(Rain, Wind=weak) + p(Rain, strong)*H(Rain, Wind=strong)
                             = (3/5)*0 + (2/5)*0 
                             = 0

        Information Gain = H(Rain) - I(Rain, Wind)
                      = 0.971 - 0
                      = 0.971
     
     Here, the attribute with maximum information gain is Wind. 

So, the decision tree built so far -

![image](https://iq.opengenus.org/content/images/2019/06/Untitled-Diagram--4-.png)

Here, when Outlook = Rain and Wind = Strong, it is a pure class of category "no". And When Outlook = Rain and Wind = Weak, it is again a pure class of category "yes".
And this is our final desired tree for the given dataset


