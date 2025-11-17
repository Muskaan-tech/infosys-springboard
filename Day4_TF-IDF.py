# Day 4: Feature Extraction using BoW and TF-IDF
 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
 
# Load preprocessed dataset (from Day 3)
df = pd.read_csv('fake_job_postings.csv')
 
# Assume we already created a 'clean_descriaption' column
texts = df['description'].fillna('').tolist()
 
# 1️⃣ Bag-of-Words
bow_vectorizer = CountVectorizer(max_features=2000)  # limit to top 2000 words
X_bow = bow_vectorizer.fit_transform(texts)
 
print("BoW shape:", X_bow.shape)
print("Sample feature names (BoW):", bow_vectorizer.get_feature_names_out()[:10])
 
# 2️⃣ TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)
 
print("\nTF-IDF shape:", X_tfidf.shape)
print("Sample feature names (TF-IDF):", tfidf_vectorizer.get_feature_names_out()[:10])
 
# 3️⃣ Compare sparsity and values
print("\nExample BoW vector (first row):")
print(X_bow[0].toarray())
 
print("\nExample TF-IDF vector (first row):")
print(X_tfidf[0].toarray())

print(df.columns)



