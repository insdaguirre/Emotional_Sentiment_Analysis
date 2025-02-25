import pandas as pd
import matplotlib.pyplot as plt


train_df = pd.read_csv("/Users/diegoaguirre/ML_Creative_Project-2/ML_Creative_Project_Submission_Folder/train.csv")
test_df = pd.read_csv("/Users/diegoaguirre/ML_Creative_Project-2/ML_Creative_Project_Submission_Folder/test.csv")


train_lengths = train_df['text'].dropna().apply(len)
test_lengths = test_df['text'].dropna().apply(len)


plt.figure(figsize=(10, 6))
plt.hist(train_lengths, bins=30, alpha=0.5, label="Train", color='blue')
plt.hist(test_lengths, bins=30, alpha=0.5, label="Test", color='orange')


plt.xlabel("Character Length")
plt.ylabel("Frequency")
plt.title("Distribution of Character Lengths in Train and Test Datasets")
plt.legend()

plt.xlim(0, 400)


plt.show()
