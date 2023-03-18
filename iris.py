# Import required libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Define features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create decision tree classifier object
clf = DecisionTreeClassifier()

# Train the decision tree classifier
clf = clf.fit(X_train, y_train)

# Get input from user for sepal-length, sepal-width, petal-length, and petal-width
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

# Create a new input array
new_input = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make a prediction for the new input
prediction = clf.predict(new_input)

# Print the predicted class
print("Predicted class:", prediction[0])

# Suggest user to take tests based on symptoms
if prediction == 'Iris-versicolor':
    print("Suggested test: allergy test")
elif prediction == 'Iris-virginica':
    print("Suggested test: chest x-ray")
else:
    print("No suggested test for this class")

# Get input from user for test results
test_result = float(input("Enter test result: "))

# Create a new input array with test result included
new_input_with_test = [[sepal_length, sepal_width, petal_length, petal_width, test_result]]

# Make a new prediction with the test result included
new_prediction = clf.predict(new_input_with_test)

# Print the new predicted class
print("New predicted class:", new_prediction[0])
