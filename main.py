import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#a)Load the "loan_old.csv" dataset
loan_old = pd.read_csv('loan_old.csv')
#b)Perform analysis on the dataset to:
#i) check whether there are missing values

missing_values = loan_old.isnull().sum()
loan_old = loan_old.drop(columns=["Loan_ID"])
if missing_values.sum() > 0:
    print("There are missing values")
    print("Total missing values:", missing_values.sum())
else:
    print("There are no missing values")

categorical_columns = [column for column in loan_old.columns if loan_old[column].dtype == 'object']
numerical_columns = [column for column in loan_old.columns if loan_old[column].dtype != 'object']
print("Categorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)

# iii) Check whether numerical features have the same scale
numerical_columns_data = loan_old.select_dtypes(include=[int, float])
range_values = numerical_columns_data.max() - numerical_columns_data.min()
for column in numerical_columns:
    print(" Ranage Of " + column+ " Column = ", max(loan_old[column]) - min(loan_old[column]))

if (range_values.iloc[0] == range_values).all():
    print("The numerical features have the same scale")
else:
    print("The numerical features do not have the same scale")

# iv) Visualize a pairplot between numerical columns
sns.pairplot(numerical_columns_data)
plt.show()
#c)
# i) Remove records containing missing values
print("Loan Old Before removing null")
print(loan_old)
loan_old = loan_old.dropna()
print("Loan Old after removing null")
print(loan_old)
# ii) Separate features and targets
X = loan_old.drop(columns=["Max_Loan_Amount", "Loan_Status"])
y_Max_Loan_Amount = loan_old["Max_Loan_Amount"]
y_Loan_Status = loan_old["Loan_Status"]
print("Features Data")
print(X)
print("Target Data")
print(y_Loan_Status)
print(y_Loan_Status)
# iii) Shuffle and split data into training and testing sets
X_train, X_test, y_Max_Loan_Amount_train, y_Max_Loan_Amount_test, y_Loan_Status_train, y_Loan_Status_test = train_test_split(
    X, y_Max_Loan_Amount, y_Loan_Status, test_size=0.3, random_state=42
)
# iv) Categorical features encoding
categorical_features = ["Gender", "Married", "Education", "Property_Area","Dependents"]
encoder_categorical_features = LabelEncoder()
for feature in categorical_features:
    X_train[feature] = encoder_categorical_features.fit_transform(X_train[feature])
    X_test[feature] = encoder_categorical_features.transform(X_test[feature])
print("Category Encode: -")
print(X_train)
# v) Categorical targets encoding
encoder_categorical_targets = LabelEncoder()
y_status_train = encoder_categorical_targets.fit_transform(y_Loan_Status_train)
y_status_test = encoder_categorical_targets.transform(y_Loan_Status_test)

# vi) Numerical features standardization
numerical_features = ["Income", "Coapplicant_Income", "Loan_Tenor", "Credit_History"]
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features].values)
X_test[numerical_features] = scaler.transform(X_test[numerical_features].values)

# d) Fit a linear regression model
LR = LinearRegression()
numeric_columns_X_train = X_train.select_dtypes(include=[np.number])
LR.fit(numeric_columns_X_train, y_Max_Loan_Amount_train)
y_Max_Loan_Amount_prediction = LR.predict(X_test[numeric_columns_X_train.columns])


#e) Evaluate the linear regression model using sklearn's R2 score.
r2 = r2_score(y_Max_Loan_Amount_test, y_Max_Loan_Amount_prediction)
print(f"Linear Regression R2 Score: {r2:.4f}")

#f) Fit a logistic regression model using gradient descent
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def initialize_parameters(n_features):
    weights = np.zeros(n_features)
    bias = 0
    return weights, bias
def gradient_descent(X, y, weights, bias, learning_rate, num_iterations):
    m = len(y)
    for _ in range(num_iterations):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    y_pred_class = np.where(y_pred > 0.5, 1, 0)
    return y_pred_class

numeric_columns_X_train = X_train.select_dtypes(include=[np.number])
numeric_columns_X_test = X_test.select_dtypes(include=[np.number])

n_features = numeric_columns_X_train.shape[1]
weights, bias = initialize_parameters(n_features)
learning_rate = 0.01
num_iterations = 1000

weights, bias = gradient_descent(numeric_columns_X_train, y_status_train, weights, bias, learning_rate, num_iterations)
y_pred = predict(numeric_columns_X_test,weights,bias)


#Write a function to calculate the accuracy of the model
def accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy
print("True : ")
print(y_status_test)
print("Prediction : ")
print(y_pred)
accuracy = accuracy(y_status_test,y_pred)
print("Accuracy:", accuracy)

#h) Load the "loan_new.csv" dataset.
loan_new = pd.read_csv('loan_new.csv')

# I)
#i) Records containing missing values are removed
loan_new = loan_new.dropna()
loan_new = loan_new.drop(columns=["Loan_ID"])

# ii) Separate features and targets
X_new = loan_new
y_Max_Loan_Amount_new = loan_new["Loan_Tenor"]
y_Loan_Status_new = loan_new["Credit_History"]

# iii) Categorical features encoding
encoder_categorical_features = LabelEncoder()
categorical_columns_new = [column for column in X_new.columns if X_new[column].dtype == 'object']
for feature in categorical_columns_new:
    encoder_categorical_features.fit(X_new[feature])
    X_new[feature] = encoder_categorical_features.fit_transform(X_new[feature])


# iv) Categorical targets encoding
encoder_categorical_targets = LabelEncoder()
y_status_new = encoder_categorical_targets.fit_transform(y_Loan_Status_new)

# v) Numerical features standardization
X_new[numerical_features] = scaler.transform(X_new[numerical_features].values)

# Print first few rows of loan_new for verification
print("First Few Rows of loan_new:")
print(loan_new.head())

#J)Use your models on this data to predict the loan amounts and status
#loan prediction

prediction_of_max_loans = LR.predict(X_new[numeric_columns_X_train.columns])
df_result = pd.DataFrame({'prediction_of_max_loans': prediction_of_max_loans})
#loan status
numeric_columns_X_new = X_new.select_dtypes(include=[np.number])
prediction_of_loan_status = predict(numeric_columns_X_new,weights,bias)
df_result['prediction_of_loan_status'] = prediction_of_loan_status

print(df_result)
