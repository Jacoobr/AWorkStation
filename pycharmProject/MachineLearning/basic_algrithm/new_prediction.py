import os
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all')
#检验数据细节
#print(news.data[0])

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
print("the target is: ")
print(news.target_names)
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
#X_test = ["bicycle"]
print("Please input the context: ")
X_test = input()
X_test = [X_test]
X_test = vec.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predict = mnb.predict(X_test)
print("the prediction is: " + news.target_names[y_predict[0]])
