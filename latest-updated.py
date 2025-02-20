import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import string

# Load datasets
df_fake = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\Fake.csv")
df_true = pd.read_csv(r"C:\Users\Rohan Nambiar\Documents\Vscode\True.csv")

# Add labels
df_fake["class"] = 0
df_true["class"] = 1

# Display total number of samples
print(f"Total fake news samples: {len(df_fake)}")
print(f"Total true news samples: {len(df_true)}")

# Remove last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10).copy()
df_true_manual_testing = df_true.tail(10).copy()

for i in range(23480, 23470, -1):
    df_fake.drop(i, axis=0, inplace=True)
for i in range(21416, 21406, -1):
    df_true.drop(i, axis=0, inplace=True)

# Label manual testing data
df_fake_manual_testing.loc[:, "class"] = 0
df_true_manual_testing.loc[:, "class"] = 1

# Combine manual testing data
df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv", index=False)

# Combine main datasets
df_merge = pd.concat([df_fake, df_true], axis=0)

# Drop unnecessary columns
df = df_merge.drop(["title", "subject", "date"], axis=1)

# Check for missing values
print("Missing values in the dataset:")
print(df.isnull().sum())

# Reset index
df.reset_index(drop=True, inplace=True)

# Text cleaning function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Apply text cleaning
df["text"] = df["text"].apply(wordopt)

# Split data into features and labels
x = df["text"]
y = df["class"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# TF-IDF Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

# Evaluate model
pred_rfc = RFC.predict(xv_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, pred_rfc))

# Function to convert numerical label to text
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_RFC = RFC.predict(new_xv_test)
    print(f"\nRFC Prediction: {output_label(pred_RFC[0])}")

# Test with user input
if __name__ == "__main__":
    news = input("Enter a news article: ")
    manual_testing(news)
