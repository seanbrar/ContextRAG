import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

text = "Natural Language Processing is evolving rapidly."
tokens = nltk.word_tokenize(text)

stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if not w.lower() in stop_words]

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]

print("Original:", tokens)
print("Filtered:", filtered_tokens)
print("Stemmed:", stemmed_tokens)
print("Lemmatized:", lemmatized_tokens)
