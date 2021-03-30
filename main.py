import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing as sk_preprocessing
from sklearn.naive_bayes import GaussianNB

# Reading the file and saving the information in a dataframe
data = pd.read_csv("Dataset/Qualitative_Bankruptcy.data.txt", sep=",", header=None)
data.columns = [
    "Industrial Risk",
    "Management Risk",
    "Financial Flexibility",
    "Credibility",
    "Competitiveness",
    "Operating Risk",
    "Class",
]

data.head()

# Transforming letters into numbers
le = LabelEncoder()
data_numbers = data[data.columns[:]].apply(le.fit_transform)
data_numbers.head()

# le = LabelEncoder()
# back = data_numbers[data.columns[:]].apply(le.inverse_transform)
# back.head()

# Choosing the Target Attribute
# Conventionally the class or target attribute is labeled as Y and the rest of the dataset is X
Y = data_numbers["Class"]
X = data_numbers.drop("Class", axis=1)

# Splitting the data
plt.hist(data_numbers["Class"], bins=2, color="orange", edgecolor="black")
plt.xlabel("Class")
plt.ylabel("Number of records")

# We train the classifier on propotionally split data. We use 70% of data to train the model and 30% to test it. We are also shuffling the data, because the data is sorted by class, which might generate biased results.
# To split the data into testing and training sets we are going to use the `train_test_split()` function from `sklearn.model_selection`.
# The `stratify` parameter is used to proportionally split according to distribution of target class.
# Documentaion: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, stratify=Y, shuffle=True
)

# Scaling the data for better accuracy and less inexpectancies
scaler = sk_preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Making of SVM (Support-Vector Machine)
# This SVC class allows us to build a kernel SVM model (linear as well as non-linear), The default value of the kernel is ‘rbf’. Why ‘rbf’, because it is nonlinear and gives better results as compared to linear.
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, Y_train)

# Predict the Test Set Results
Y_pred = classifier.predict(X_test)

# Model Accuracy for SVM
# Confusion Matrices
# To build the confusion matrix we need to get the predictions of the trained model on the testing dataset i.e. `X_test`
cm = confusion_matrix(Y_test, Y_pred)
accuracy_score(Y_test, Y_pred)
print("Non-linear SVM error rate =", 1 - accuracy_score(Y_test, Y_pred))


# Decision Tree
# The Classifer is built using the `DecisionTreeClassifier()` method from `sklearn.tree`. The parameter `criterion` is used to specify the type of decision tree.
clf = tree.DecisionTreeClassifier(criterion="gini", max_leaf_nodes=None)

# Training the classifer
# We will train the classifier on the traing data i.e. `X_train`, `Y_train`.
clf.fit(X_train, Y_train)

# Model Accuracy
# Confusion Matrices
# To build the confusion matrix we need to get the predictions of the trained model on the testing dataset i.e. `X_test`
Y_predictions = clf.predict(X_test)
Y_predictions2 = clf.predict(X_train)

# The confusion matrix for our model is built using `confusion_matrix` from `sklearn.metrics`.
cm = confusion_matrix(Y_predictions, Y_test)
cm2 = confusion_matrix(Y_predictions2, Y_train)

# To get the accuracy we need to get the sum of (true positives   true negatives) / total
# This metric gives overall how often the classifer is correct.
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

# Decision tree test set error rate
test_accuracy = accuracy(cm)
print("Decision tree test set error rate =", 1 - test_accuracy)

# Decision tree Training set error rate
training_accuracy = accuracy(cm2)
print("Decision tree training set error rate =", 1 - training_accuracy)

# Tree Visualization
# The Tree is viusalized using `sklearn.tree.plot_tree()` This is plotted on a matplotlib figure and matplotlib commands can be used to style it.
# The overlap of nodes is a bug present in the plot_tree() function when plotting some kinds of trees. To view the hidden nodes try changing `figsize` to adapt the plot to your monitor resolution.
plt.figure(figsize=(15, 10))
tree.plot_tree(
    clf,
    filled=True,
    fontsize=10,
    max_depth=None,
    feature_names=list(X.columns),
    class_names=True,
)
# plt.show() 


# Building shorter trees - 3 nodes
clf_3 = tree.DecisionTreeClassifier(criterion="gini", max_leaf_nodes=3)
clf_3.fit(X_train, Y_train)

# Confusion Matrices for 3 node tree
Y_3_predictions = clf_3.predict(X_test)
Y_3_predictions2 = clf_3.predict(X_train)

# The confusion matrix for our model is built using `confusion_matrix` from `sklearn.metrics`.
cm_3 = confusion_matrix(Y_3_predictions, Y_test)
cm2_3 = confusion_matrix(Y_3_predictions2, Y_train)

# 3 node tree test set error rate
test_accuracy_3 = accuracy(cm_3)
print("3 node tree test set error rate =", 1 - test_accuracy_3)

# 3 node tree training set error rate
training_accuracy_3 = accuracy(cm2_3)
print("3 node tree training set error rate =", 1 - training_accuracy_3)

# Plotting 3 node tree
plt.figure(figsize=(12, 10))
tree.plot_tree(
    clf_3,
    filled=True,
    fontsize=10,
    max_depth=None,
    feature_names=list(X.columns),
    class_names=True,
)
# plt.show()



# Gaussian Naive Bayes
NBclassifier = GaussianNB()
NBclassifier.fit(X_train, Y_train)
y_pred_Gaussian = classifier.predict(X_test)
cm_Gaussian = confusion_matrix(y_pred_Gaussian, Y_test)
print("Gaussian error rate =", 1 - accuracy(cm_Gaussian))


print("----------------------------")

# Prediction for custom data set input
predict_this = []
sample = []

predict_this.append(sample)

print("Custom data input\nAttribute Information: (Average = 0, Negative = 1, Positive = 2, B-Bankruptcy, NB-Non-Bankruptcy)")

questions = ["Industrial Risk", "Management Risk", "Financial Flexibility", "Credibility", "Competitiveness", "Operating Risk"]

for i in range(6):
  value = int(input(questions[i] + ": "))
  if value == 0 or value == 1 or value == 2:
    sample.append(value)
  else:
    raise ValueError("Value not allowed. Try 0 for Average, 1 for Negative, or 2 for Positive.")
    
  
if clf.predict(predict_this) == 1:
    print("Non-Bankruptcy")
else:
    print("Bankruptcy")
