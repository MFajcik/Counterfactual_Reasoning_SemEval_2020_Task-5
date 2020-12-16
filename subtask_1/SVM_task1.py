# SVM baseline for SemEval-2020 Subtask-1

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
np.random.seed(500)

print(">> Read data...")
path = "Subtask-1-master/train.csv"
corpus = pd.read_csv(path, encoding='utf-8')
percent = 0.3 	# 0.3 for testing
print("File: %s" % path)

corpus['sentence'].dropna(inplace=True)
corpus['sentence'] = [sent.lower() for sent in corpus['sentence']]
corpus['sentence'] = [word_tokenize(word) for word in corpus['sentence']]
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for index, entry in enumerate(corpus['sentence']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    corpus.loc[index, 'sentence_final'] = str(Final_words)
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['sentence_final'], corpus['gold_label'], test_size=percent)

print(">> Feature generation...")
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(corpus['sentence_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
print(Train_X_Tfidf)
print(">> SVM classifier....")
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
gnb = GaussianNB()
predictions_nb=gnb.fit(Train_X_Tfidf.toarray(), Train_Y).predict(Test_X_Tfidf.toarray())
#print(predictions_SVM)
#print(predictions_SVM.shape)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(64, 6), random_state=1)
clf.fit(Train_X_Tfidf, Train_Y)
predictions_mlp=clf.predict(Test_X_Tfidf)
print(predictions_mlp.shape)
predictions_M=np.zeros((3000,),dtype=int)
print(predictions_M.shape)
print(predictions_M)
print(predictions_mlp)
print("Majority Precision Score -> ", accuracy_score(Test_Y, predictions_M) * 100)
print("Majority Recall Score -> ", recall_score(Test_Y, predictions_M) * 100)
print("Majority F1 Score -> ", f1_score(Test_Y, predictions_M) * 100)

# print("SVM Accuracy Score -> ", accuracy_score(Test_Y, predictions_SVM) * 100)
print("SVM Precision Score -> ", precision_score(Test_Y, predictions_SVM) * 100)
print("SVM Recall Score -> ", recall_score(Test_Y, predictions_SVM) * 100)
print("SVM F1 Score -> ", f1_score(Test_Y, predictions_SVM) * 100)

print("NB Precision Score -> ", precision_score(Test_Y, predictions_nb) * 100)
print("NB Recall Score -> ", recall_score(Test_Y, predictions_nb) * 100)
print("NB F1 Score -> ", f1_score(Test_Y, predictions_nb) * 100)

print("MLP Precision Score -> ", precision_score(Test_Y, predictions_mlp) * 100)
print("MLP Recall Score -> ", recall_score(Test_Y, predictions_mlp) * 100)
print("MLP F1 Score -> ", f1_score(Test_Y, predictions_mlp) * 100)
