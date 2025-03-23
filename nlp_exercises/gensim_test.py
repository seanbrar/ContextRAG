from gensim import corpora, models
import gensim

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Sample data: documents split into words
documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
]

stop_words = set(stopwords.words("english"))

# Tokenization and removing stop words
texts = [
    [word for word in document.lower().split() if word not in stop_words]
    for document in documents
]

# Creating a dictionary
dictionary = corpora.Dictionary(texts)

# Creating the corpus
corpus = [dictionary.doc2bow(text) for text in texts]

# Applying LDA
ldamodel = gensim.models.ldamodel.LdaModel(
    corpus, num_topics=3, id2word=dictionary, passes=15
)

# Print the topics
for idx, topic in ldamodel.print_topics(-1):
    print("Topic: {}\nWords: {}".format(idx, topic))
