
# Day 3: Text Cleaning and Preprocessing
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
# Download resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')
 
# Load dataset (same as Day 2)
df = pd.read_csv('fake_job_postings.csv', low_memory=False)

 
# Define text cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # 4. Remove punctuation and numbers
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)
    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # 6. Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)
 
# Apply cleaning to key text columns
df['clean_description'] = df['description'].apply(clean_text)
 
# Show before and after
print("Original Text:\n\n", df['description'].iloc[1][:300])
print("\nCleaned Text:\n", df['clean_description'].iloc[1][:300])
 
# Check for any remaining issues
print("\nExample of Cleaned Data:")
print(df[['description', 'clean_description']].head(3))

