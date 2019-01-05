''' 
    Program extracts comments from Reddit and labels negative/offensive comments 
    as well as those otherwise constituting Cyberbulling

    Models are trained using naive Bayes with the bag_of_words approach and
    the term frequency - inverse document frequency (tfidf) approach, as well 
    as somewhat custom approach using support vector machine
'''

# Data Processing Tools
import praw
import pickle
import pandas
import numpy

# Machine Learning Tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Natural Language Tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re # regular expressions for string comparison 

# The classifyer
class CyberbullyingDetectionEngine:
    def __init__(self):
        self.corpus = None
        self.tags = None
        self.lexicons = None
        self.vectorizer = None
        self.model = None
        self.metrics = None

    # Extracts and vectorizes text features
    class CustomVectorizer:
        def __init__(self, lexicons):
            self.lexicons = lexicons
        
        # returns a numpy array of word vectors
        def transform(self, corpus):
            word_vectors = []
            for text in corpus:
                features = []
                for k, v in self.lexicons.items():
                    features.append(len([w for w in word_tokenize(text) if w in v]))
                word_vectors.append(features)
            return numpy.array(word_vectors)
        
    # takes a list of strings, removes stopwords, converts to lowercase, removes 
    # non-alphanumeric characters, and stems words (reduces words to roots to prevent duplication)
    def _simplify (self, corpus):
        stop_words = set(stopwords.words('english'))
        stemmer = SnowballStemmer('english')
            
        def clean(text):
            text = re.sub('[^a-zA-Z0-9]', ' ', text) # use regular expression to substitue gap punctuation, etc.
            words = [stemmer.stem(w) for w in word_tokenize(text.lower()) if w not in stop_words]
            return " ".join(words)
            
        return [clean(text) for text in corpus]
    
    # Takes in a path to a text file and returns a set containing every word in file
    def _get_lexicon(self, path):
        words = set()
        with open(path) as file:
            for line in file:
                words.update(line.strip().split(' '))
        return words

    # returns dictionary of metrics describing testing data
    def _model_metrics (self, features, tags):
        tp = 0
        fp = 0
        tn = 0
        fn = 0    

        predictions = self.model.predict(features)

        for r in zip(predictions, tags):
            if(r[0] == 1 and r[1] == 1):
                tp += 1
            elif(r[0] == 1 and r[1] == 0):
                fp += 1
            elif (r[0] == 0 and r[1] == 0):
                tn += 1
            else:
                fn += 1
          
        precision = tp/(tp+fp) # precision = true positives/(all positive tags)
        recall = tp/(tp+fn) # recall = true positives/ (all correct tags)
        
        # return dictionary of metrics
        return {
            'precision': precision,
            'recall': recall,
            'f1': (2 * precision * recall) / (precision + recall)
        }
        
    # Loads and extracts a tagged corpus (pickled pandas dataframe, corpus column name, tag column name)
    def load_corpus (self, path, corpus_col, tag_col):
        data  = pandas.read_pickle(path)
        group = data [[corpus_col, tag_col]].values
        self.corpus = [row[0] for row in group]
        self.tags = [row[1] for row in group]
        
    # Loads a set of words from a txt file
    def load_lexicon(self, fname):
        if self.lexicons == None:
            self.lexicons = {}
            
        self.lexicons[fname] = self._get_lexicon('./data/'+fname+'.txt')
        
    # Loads a ml model, its feature vectorizer, and its performance metrics
    def load_model(self, model_name):
        self.model = pickle.load(open('./models/'+model_name+'_ml_model.pkl', 'rb'))
        self.vectorizer = pickle.load(open('./models/'+model_name+'_vectorizer.pkl', 'rb'))
        self.metrics = pickle.load(open('./models/'+model_name+'_metrics.pkl', 'rb'))
   
   # Training using bag-of-words model   
    def train_using_bow (self):
        corpus = self._simplify(self.corpus)
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(corpus)

        bag_of_words = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(bag_of_words, self.tags, test_size = 0.2, stratify=self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)

    # Training using tf-idf weighted word counts as features
    def train_using_tfidif(self):
        corpus = self._simplify(self.corpus)
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(corpus)

        word_vectors = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(word_vectors, self.tags, test_size = 0.2, stratify=self.tags)

        self.model = MultinomialNB()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)
    
    # Training using custom feature extraction approach with a support
    # vector machine
    def train_using_custom(self):
        corpus = self._simplify(self.corpus)
        self.vectorizer = self.CustomVectorizer(self.lexicons)

        word_vectors = self.vectorizer.transform(corpus)
        x_train, x_test, y_train, y_test = train_test_split(word_vectors, self.tags, test_size = 0.2, stratify=self.tags)

        self.model = SVC()
        self.model.fit(x_train, y_train)

        self.metrics = self._model_metrics(x_test, y_test)

    # Returns a dictionary of model performance metrics
    def evaluate(self):
        return self.metrics

    # Saves model for future use
    def save_models (self, model_name):
        pickle.dump(self.model, open('./models/'+model_name+'_ml_model.pk', 'wb'))
        pickle.dump(self.vectorizer, open('./models/'+model_name+'_vectorizer.pkl', 'wb'))
        pickle.dump(self.metrics, open('./models/'+model_name+'_metrics.pkl', 'wb'))
    
    # Returns predictions based on a text corupus
    def predict(self, corpus):
        x = self.vectorizer.transform(self._simplify(corpus))
        return self.model.predict(x)

# Testing code
if __name__ == '__main__':
    # Get Reddit objects
    reddit = praw.Reddit (
        client_id = 'Kb_ngMdlcLrIlA',
        client_secret = 'z-pTnxd0Bj5SYvmn8UbDfrvYPRg',
        user_agent = 'testing'
    )

    # Get data
    new_comments = reddit.subreddit('TwoXChromosomes').comments(limit=1000)
    queries = [comment.body for comment in new_comments]

    # Train using bag-of-words, save and display it
    engine = CyberbullyingDetectionEngine()
    engine.load_corpus('./data/final_labelled_data.pkl', 'tweet', 'class')

    engine.train_using_bow()
    print(engine.evaluate())
    print(engine.predict(queries))
    engine.save_models('bow_2')


