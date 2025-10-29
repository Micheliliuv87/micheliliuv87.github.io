---
layout: post
title: Practical Applications of the Bag of Words Model
date: 2025-01-05 21:30:21
description: Application of Bag of words, examples of TF-IDF, Text Classification, and Sentiment Analysis
tags: Tutorials
categories: Text Processing
citation: true
toc:
  sidebar: left
authors:
  - name: Qirui(Micheli) Liu
    url: "https://micheliliuv87.github.io/"
    affiliations: 
    name: Emory University ISOM
---

## **Introduction & Recap**

In our previous post, we explored the **theoretical foundations** of the Bag of Words (BoW) model—how it converts unstructured text into numerical representations by counting word frequencies while disregarding word order and grammar. This fundamental technique has proven remarkably effective across various **Natural Language Processing (NLP) tasks** despite its simplicity .

Now, let's transition from theory to practice. In this hands-on guide, we'll implement the Bag of Words model for several real-world applications including text classification, sentiment analysis, and TF-IDF visualization. We'll work with Python libraries like `scikit-learn`, `NLTK`, and `matplotlib` to build working examples you can adapt for your own projects.

## **Text Classification with BoW: Spam Detection Example**

One of the most common applications of Bag of Words is **text classification**, where we categorize documents into predefined classes. Let's build a spam detection system that classifies messages as "spam" or "not spam."

#### **Step-by-Step Implementation**

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# download NLTK sample data
nltk.download('stopwords')
nltk.download('punkt')

# sample dataset (in practice, you'd use a larger dataset)
data = {
    'text': [
        'Win a free iPhone now! Click here to claim your prize.',
        'Your package has been delivered. Track your shipment.',
        'Congratulations! You won a $1000 gift card. Reply to claim.',
        'Meeting scheduled for tomorrow at 10 AM in conference room.',
        'Urgent! Your account needs verification. Update immediately.',
        'The quarterly report is attached for your review.'
    ],
    'label': ['spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam']
}

df = pd.DataFrame(data)

# text preprocessing function
def preprocess_text(text):
    # remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^A-Za-z]', ' ', text.lower())

    #tokenize
    words = nltk.word_tokenize(text)

    #remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    #apply stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

# apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# create Bag of Words representation
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['processed_text']).toarray()
y = df['label']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train a Naive Bayes classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# make predictions and evaluate accuracy
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
```

#### **How BoW Helps in Classification**

In this spam detection example, the Bag of Words model identifies **characteristic word patterns** in spam versus legitimate messages . Spam messages often contain words like "win," "free," "prize," and "urgent," while legitimate messages use more neutral language. By converting these word patterns into numerical features, we enable machine learning algorithms to learn the distinguishing characteristics of each category.

The `CountVectorizer` from scikit-learn handles the heavy lifting of creating our BoW representation . The `max_features` parameter ensures we only consider the top 1000 most frequent words, preventing excessively high-dimensional data.

## **Sentiment Analysis with BoW**

Another powerful application of Bag of Words is **sentiment analysis**—determining whether a piece of text expresses positive, negative, or neutral sentiment. Let's analyze movie reviews.

#### **Implementation Code**

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Sample movie reviews dataset
reviews = [
    'This movie was absolutely fantastic! Great acting and plot.',
    'Terrible waste of time. Poor acting and boring story.',
    'Loved the cinematography and character development.',
    'The worst movie I have ever seen in my life.',
    'Brilliant performance by the lead actor. Highly recommended.',
    'Mediocre at best. Nothing special about this film.',
    'An outstanding masterpiece that kept me engaged throughout.',
    'Poor direction and weak screenplay. Very disappointing.'
]

sentiments = ['positive', 'negative', 'positive', 'negative',
              'positive', 'negative', 'positive', 'negative']

# Create BoW model with unigrams and bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(reviews)

# Train a classifier
classifier = MultinomialNB()
classifier.fit(X, sentiments)

# Test with new reviews
test_reviews = [
    'The acting was good but the story was weak',
    'Amazing movie with fantastic performances'
]

test_vectors = vectorizer.transform(test_reviews)
predictions = classifier.predict(test_vectors)

for review, sentiment in zip(test_reviews, predictions):
    print(f"Review: '{review}' -> Sentiment: {sentiment}")
```

#### **Understanding the Output**

This example introduces an important enhancement: **ngrams** . By setting `ngram_range=(1, 2)`, we consider both single words (unigrams) and pairs of consecutive words (bigrams). This helps capture phrases like "absolutely fantastic" or "poor acting" that carry more nuanced sentiment than individual words.

The `MultinomialNB` classifier is particularly well-suited for text classification with discrete features like word counts . It efficiently learns the probability distributions of words for each sentiment class.

## **TF-IDF: Enhancing Bag of Words**

While simple word counts are useful, they have a limitation: common words that appear frequently across all documents may dominate the analysis. **TF-IDF (Term Frequency-Inverse Document Frequency)** addresses this by weighting words based on their importance .

#### **TF-IDF Calculation and Visualization**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# sample documents for TF-IDF demonstration
documents = [
    "Natural language processing with Python is fascinating and powerful",
    "Python is a great programming language for data science",
    "I enjoy learning new things about NLP and Python programming",
    "Machine learning and artificial intelligence are changing the world",
    "Deep learning models require extensive computational resources",
    "Data scientists use Python for machine learning projects"
]

# calculate TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# get feature names and TF-IDF scores
feature_names = tfidf_vectorizer.get_feature_names_out()
dense_matrix = tfidf_matrix.todense()

# display TF-IDF scores for first document
first_doc = dense_matrix[0].tolist()[0]
word_scores = sorted(zip(feature_names, first_doc),
                    key=lambda x: x[1], reverse=True)[:5]

print("Top terms in first document:")
for word, score in word_scores:
    print(f"{word}: {score:.4f}")

#dimension reduction for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(dense_matrix)

# create visualization
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', s=100, alpha=0.7)

#ddd document labels
for i, txt in enumerate([f"Doc {i+1}" for i in range(len(documents))]):
    plt.annotate(txt, (pca_result[i, 0], pca_result[i, 1]),
                 xytext=(5, 5), textcoords='offset points')

plt.title('TF-IDF Document Visualization using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True, alpha=0.3)
plt.show()
```

#### **TF-IDF Intuition**

TF-IDF balances two factors :

- **Term Frequency (TF)**: How often a word appears in a specific document
- **Inverse Document Frequency (IDF)**: How rare the word is across all documents

The product of these two values gives higher weight to words that are frequent in a specific document but rare in the overall collection. This effectively identifies words that are characteristic of each document.

<img src="https://i.ytimg.com/vi/zLMEnNbdh4Q/maxresdefault.jpg" alt="TF-IDF Visualization" style="display:block; margin:0 auto; max-width:100%; height:auto; max-height:480px;" />

_TF-IDF helps identify the most distinctive words in documents, pushing similar documents closer together in vector space ._

## **Advanced Techniques & Best Practices**

#### **Feature Engineering with N-grams**

As mentioned in the sentiment analysis example, n-grams can significantly improve model performance:

```python
# comparing different n-gram ranges
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
trigram_vectorizer = CountVectorizer(ngram_range=(1, 3))

# each approach captures different linguistic patterns
```

Bigrams and trigrams help preserve contextual information that single words might lose . For example, the phrase "not good" has a very different meaning than the individual words "not" and "good" considered separately.

#### **Handling Large Vocabularies with Feature Hashing**

For extremely large datasets, the BoW representation can become computationally challenging. **Feature hashing** (or the "hash trick") provides a memory-efficient alternative :

```python
from sklearn.feature_extraction.text import HashingVectorizer

# using hashing vectorizer for large datasets
hash_vectorizer = HashingVectorizer(n_features=1000, alternate_sign=False)
X_hash = hash_vectorizer.transform(documents)
```

## **Conclusion**

The Bag of Words model, while simple, remains remarkably powerful for practical text analysis tasks. Through our implementations, we've seen how BoW enables:

- **Text classification** by identifying characteristic word patterns in different categories
- **Sentiment analysis** by capturing emotional language through word frequencies
- **Enhanced analysis with TF-IDF** by weighting words based on their distinctiveness

The true power of Bag of Words lies in its **versatility and interpretability**. Unlike more complex deep learning models, BoW features are easily understandable and can provide valuable insights into what the model is learning.

While modern approaches like word embeddings and transformer models have their place for complex NLP tasks, Bag of Words remains an excellent starting point for most text analysis projects—offering a compelling balance of simplicity, interpretability, and effectiveness.

_All code examples in this post are designed to be runnable with Python 3.6+ and standard data science libraries (pandas, scikit-learn, NLTK, matplotlib)._

---

_This practical guide builds upon the theoretical foundations established in our previous post about the Bag of Words model. Implement these techniques as a starting point for your text analysis projects, then experiment with different preprocessing approaches and parameter tuning to optimize for your specific use cases._

## **References**:

1. [Python Bag of Words Models](https://www.datacamp.com/tutorial/python-bag-of-words-model)
2. [Visualizing TF-IDF Scores: A Comprehensive Guide to Plotting a Document TF-IDF 2D Graph](https://www.geeksforgeeks.org/machine-learning/visualizing-tf-idf-scores-a-comprehensive-guide-to-plotting-a-document-tf-idf-2d-graph/)
3. [Text classification using the Bag Of Words Approach with NLTK and Scikit Learn](https://medium.com/swlh/text-classification-using-the-bag-of-words-approach-with-nltk-and-scikit-learn-9a731e5c4e2f)
4. [Vector Visualization: 2D Plot your TF-IDF with PCA](https://medium.com/@GeoffreyGordonAshbrook/vector-visualization-2d-plot-your-tf-idf-with-pca-83fa9fccb1d)
5. [Bag of words (BoW) model in NLP](https://www.geeksforgeeks.org/nlp/bag-of-words-bow-model-in-nlp/)
6. [A friendly guide to NLP: Bag-of-Words with Python example](https://www.analyticsvidhya.com/blog/2021/08/a-friendly-guide-to-nlp-bag-of-words-with-python-example/)
7. [3 Analyzing word and document frequency: tf-idf](https://www.tidytextmining.com/tfidf)
8. [Analyzing Documents with TF-IDF](https://programminghistorian.org/en/lessons/analyzing-documents-with-tfidf)
9. [A Gentle Introduction to the Bag-of-Words Model](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
10. [Bag of Words: Approach, Python Code, Limitations](https://blog.quantinsti.com/bag-of-words/)
