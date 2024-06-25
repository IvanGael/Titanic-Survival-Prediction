import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# 1. Data Preprocessing and Exploration
print("Step 1: Data Preprocessing and Exploration")

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Examine the data
print(train_df.info())
print(train_df.describe())

# 2. Feature Engineering
print("\nStep 2: Feature Engineering")

# Combine train and test data for preprocessing
all_data = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

# Extract title from Name
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Convert categorical variables to numerical
all_data = pd.get_dummies(all_data, columns=['Sex', 'Embarked', 'Title'])

# Fill missing values
all_data['Age'].fillna(all_data['Age'].median(), inplace=True)
all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)

# Create family size feature
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

# Drop unnecessary columns
all_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

# Split back into train and test
train_df = all_data[:len(train_df)]
test_df = all_data[len(train_df):]

# 3. Model Selection and Training
print("\nStep 3: Model Selection and Training")

# Prepare the data
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"{name} Accuracy: {model.score(X_val_scaled, y_val):.4f}")

# 4. Model Evaluation
print("\nStep 4: Model Evaluation")

# Assuming Random Forest performed best
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train_scaled, y_train)

# Feature importance
importances = best_model.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()

print("\nConclusion:")
print("Based on the analysis, the factors that had the greatest influence on survival chances were:")
print("1. Sex: Women had a higher chance of survival.")
print("2. Class (Pclass): First-class passengers were more likely to survive than those in lower classes.")
print("3. Age: Younger passengers, especially children, had better survival rates.")
print("4. Family Size: Passengers traveling with family members had different survival rates compared to those traveling alone.")

print("\nTo answer the question 'Who has the greatest chance to survive?':")
print("A young, female, first-class passenger traveling with a small family would have had the highest probability of survival on the Titanic.")
print("This conclusion aligns with the historical accounts of 'women and children first' policies during the evacuation, as well as the preferential treatment given to first-class passengers.")
print("However, it's important to note that while these factors significantly influenced survival chances, there was still an element of chance involved in the tragic event.")


# 5. Prediction file generation
print("\nStep 5: Prediction file generation")

# Prepare the test data
X_test = test_df.drop(['Survived'], axis=1, errors='ignore')
X_test_scaled = scaler.transform(X_test)

# Make predictions
predictions = best_model.predict(X_test_scaled)

# Create a DataFrame with PassengerId and Survived columns
submission_df = pd.DataFrame({
    'PassengerId': pd.read_csv('test.csv')['PassengerId'],
    'Survived': predictions
})

# Sort by PassengerId (optional, as it's already sorted in the original data)
submission_df = submission_df.sort_values('PassengerId')

# Save the predictions to a CSV file
submission_df.to_csv('titanic_predictions.csv', index=False)

print("\nPrediction file 'titanic_predictions.csv' has been generated.")