# coding:utf-8
'''
参考网址 https://blog.csdn.net/abcjennifer/article/details/23615947
'''

import os
from sklearn.datasets import fetch_20newsgroups
import time
# news = fetch_20newsgroups(subset='all')
#
# # 检验数据细节
# print(news.DESCR)
#
# from sklearn.cross_validation import train_test_split
# # 使用train_test_split划分数据集，用随机数种子random_state采样（test_size=）25%的数据作为测试数据
# X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# print("the target is: ")
# print(news.target_names)
#
# ##使用k-nn分类器对20newsgroup进行类别预测
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
#
# #对训练和测试特征数据进行标准化
# ss = StandardScaler()
# # print(X_train[0])
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)
# #对测试数据集进行预测
# knc = KNeighborsClassifier()
# knc.fit(X_train, y_train)
# y_predict = knc.predict(X_test)
#
# # #X_test = ["bicycle"]
# # print("Please input the context: ")
# # X_test = input()
# # X_test = [X_test]
# # X_test = vec.transform(X_test)
#
# ##模型评估
#
# print('The accuracy of K-Nearest Neighbor Classifier is', knc.score(X_test, y_test))
#
# ##使用sklearn.metrics的classification_report对预测结果作进一步分析
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_predict, target_names=news.target_names))
##all categories
#newsgroup_train = fetch_20newsgroups(subset='train')
##part categories
#categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x']

# 分别获取训练集合测试集
newsgroup_train = fetch_20newsgroups(subset='train')
newsgroup_test = fetch_20newsgroups(subset='test')

# 特征提取
from sklearn.feature_extraction.text import HashingVectorizer

vectorizer = HashingVectorizer(stop_words='english', non_negative=True, n_features=10000)
fea_train = vectorizer.fit_transform(newsgroup_train.data)
fea_test = vectorizer.fit_transform(newsgroup_test.data)

# from sklearn.datasets import fetch_20newsgroups_vectorized
#
# tfidf_train_3 = fetch_20newsgroups_vectorized(subset='train')
# tfidf_test_3 = fetch_20newsgroups_vectorized(subset='test')

from sklearn import metrics
# def calculate_result(actual, pred):
#     m_precision = metrics.precision_score(actual, pred)
#     m_recall = metrics.recall_score(actual, pred)
#
#     # print('predict info:')
#     # print('precision:{0:.3f}'.format(m_precision))
#     # print('recall:{0:0.3f}'.format(m_recall))
#     # print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, pred)))

##KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knnclf = KNeighborsClassifier() #k=5
tic = time.time()   # calculate time
knnclf.fit(fea_train, newsgroup_train.target)
pred = knnclf.predict(fea_test)
toc = time.time()
print('the train and predict cost: ', (toc-tic), 's')
print(pred)
#calculate_result(newsgroup_test.target, pred)
precision = metrics.classification_report(newsgroup_test.target, pred)
print(precision)































