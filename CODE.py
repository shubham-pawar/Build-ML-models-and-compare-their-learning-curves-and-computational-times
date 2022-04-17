#Importing all required libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import learning_curve
from matplotlib import pyplot
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.dummy import DummyRegressor
import scipy

ohe = OneHotEncoder(sparse=False)
a = [["red","medium","circle"],["blue","large","square"],["green","small","triangle"]]
ohe.fit(a)
print(ohe.categories_,"\n")
b = [["green","small","circle"],["blue","medium","square"]]
new_b = ohe.transform(b)
print(new_b)
print("--------------------------------------------------------------------------------------------------------")
#Fetch dataset from openml #Another-Dataset-on-used-Fiat-500-(1538-rows)
lis = datasets.fetch_openml(data_id=43828) 
print(lis.data.info())

#print all distinct values of nominal features
print("Districe values of model ",lis.data["model"].unique())
print("--------------------------------------------------------------------------------------------------------")
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False), [0])], remainder="passthrough")

new_data = ct.fit_transform(lis.data)
print(ct.get_feature_names())

lis_new_data = pd.DataFrame(new_data, columns = ct.get_feature_names(), index = lis.data.index)

print(lis_new_data.info())

print("Target\n",lis.target)
print("--------------------------------------------------------------------------------------------------------")
#Decision Tree (DecisionTreeRegressor)
print("DecisionTreeRegressor start\n")
dt = tree.DecisionTreeRegressor()
parameters=[{"min_samples_leaf":[2,4,6,8,10]}]
tuned_dtc = model_selection.GridSearchCV(dt, parameters, cv=10)
scores_dt = model_selection.cross_validate(tuned_dtc, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
print(scores_dt["test_score"])
rmse = 0-scores_dt["test_score"]
##print("DT RMSE mean--> ",rmse.mean())

train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(tuned_dtc, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
DT_statsScore = 0-test_scores
print("DT RMSE with increasing amount of training data:\n ",DT_statsScore)
print("DT RMSE last point--> ",DT_statsScore[4].mean())
DT_statsScore = DT_statsScore.mean(axis=1)

print("DT Training Time--> ",fit_times[4].mean())
print("DT Testing Time--> ",score_times[4].mean())

#print(DT_statsScore.mean(axis=1))
#print(train_sizes)
pyplot.plot(train_sizes, DT_statsScore)
pyplot.xlabel("Number of training examples")
pyplot.ylabel("RMSE")
pyplot.show()
print("\n")
print("--------------------------------------------------------------------------------------------------------")

#K-nearest Neighbours (KNearestNeighborsRegressor) n_neighbors=5
print("KNearestNeighborsRegressor start\n")
neigh = KNeighborsRegressor()
parameters=[{"n_neighbors":(1,10,1)}]
tuned_knn = model_selection.GridSearchCV(neigh, parameters, cv=10)
scores_knn = model_selection.cross_validate(tuned_knn, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
print(scores_knn["test_score"])
rmse = 0-scores_knn["test_score"]
#print("KNN rmse mean--> ", rmse.mean())

train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(tuned_knn, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
KNN_statsScore = 0-test_scores
print("KNN RMSE with increasing amount of training data:\n ",KNN_statsScore)
print("KNN RMSE last point--> ",KNN_statsScore[4].mean())
KNN_statsScore = KNN_statsScore.mean(axis=1)

print("KNN Training Time--> ",fit_times[4].mean())
print("KNN Testing Time--> ",score_times[4].mean())

#print(KNN_statsScore.mean(axis=1))
#print(train_sizes)
pyplot.plot(train_sizes,  KNN_statsScore)
pyplot.xlabel("Number of training examples")
pyplot.ylabel("RMSE")
pyplot.show()
print("\n")
print("--------------------------------------------------------------------------------------------------------")

#LinearRegression
print("LinearRegression start\n")
lr = LinearRegression()
scores = model_selection.cross_validate(lr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
print(scores["test_score"])
rmse = 0-scores["test_score"]
print("LR rmse mean--> ",rmse.mean())

train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(lr, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
LR_statsScore = 0-test_scores
print("LR RMSE with increasing amount of training data:\n ",LR_statsScore)
print("LR RMSE last point--> ",LR_statsScore[4].mean())
LR_statsScore = LR_statsScore.mean(axis=1)

print("LR Training Time--> ",fit_times[4].mean())
print("LR Testing Time--> ",score_times[4].mean())

#print(LR_statsScore.mean(axis=1))
#print(train_sizes)
pyplot.plot(train_sizes,  LR_statsScore)
pyplot.xlabel("Number of training examples")
pyplot.ylabel("RMSE")
pyplot.show()
print("\n")
print("--------------------------------------------------------------------------------------------------------")

#SVM(Support Vector Machine regressor) - C=1.0, epsilon=0
print("Support Vector Machine regressor start\n")
svm = SVR()
scores_svm = model_selection.cross_validate(svm, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
print(scores_svm["test_score"])
rmse = 0-scores_svm["test_score"]
#print("SVM rmse mean--> ", rmse.mean())

train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(svm, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
SVM_statsScore = 0-test_scores
print("SVM RMSE with increasing amount of training data:\n ",SVM_statsScore)
print("SVM RMSE last point--> ",SVM_statsScore[4].mean())
SVM_statsScore = SVM_statsScore.mean(axis=1)

print("SVM Training Time--> ",fit_times[4].mean())
print("SVM Testing Time--> ",score_times[4].mean())

pyplot.plot(train_sizes,  SVM_statsScore)
pyplot.xlabel("Number of training examples")
pyplot.ylabel("RMSE")
pyplot.show()
print("\n")
print("--------------------------------------------------------------------------------------------------------")

#BaggingRegressor(Bagged decision tree regressor)
print("Bagged decision tree regressor start\n")
br = BaggingRegressor()
scores_br = model_selection.cross_validate(br, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
print(scores_br["test_score"])
rmse = 0-scores_br["test_score"]
#print("Bagging rmse mean--> ", rmse.mean())

train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(br, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
BR_statsScore = 0-test_scores
print("BR RMSE with increasing amount of training data:\n ",BR_statsScore)
print("BR RMSE last point--> ",BR_statsScore[4].mean())
BR_statsScore = BR_statsScore.mean(axis=1)

print("BR Training Time--> ",fit_times[4].mean())
print("BR Testing Time--> ",score_times[4].mean())

pyplot.plot(train_sizes,  BR_statsScore)
pyplot.xlabel("Number of training examples")
pyplot.ylabel("RMSE")
pyplot.show()
print("\n")
print("--------------------------------------------------------------------------------------------------------")

#DummyRegressor
print("DummyRegressor start\n")
dr = DummyRegressor()
scores_dr = model_selection.cross_validate(dr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
print(scores_dr["test_score"])
rmse = 0-scores_dr["test_score"]
#print("Dummy rmse mean--> ", rmse.mean())

train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(dr, lis_new_data, lis.target, train_sizes=[0.2, 0.4, 0.6, 0.8, 1], cv=10,return_times=True,scoring="neg_root_mean_squared_error",shuffle=True,random_state=0)
DR_statsScore = 0-test_scores
print("DR RMSE with increasing amount of training data:\n ",DR_statsScore)
print("DR RMSE last point--> ",DR_statsScore[4].mean())
DR_statsScore = DR_statsScore.mean(axis=1)
print("DR Training Time--> ",fit_times[4].mean())
print("DR Testing Time--> ",score_times[4].mean())

pyplot.plot(train_sizes,  DR_statsScore)
pyplot.xlabel("Number of training examples")
pyplot.ylabel("RMSE")
pyplot.show()
print("\n")
print("--------------------------------------------------------------------------------------------------------")

print("\n------------------Compare best model with rest models-----------------")
print("LR vs DT ",scipy.stats.ttest_rel(LR_statsScore, DT_statsScore))
print("LR vs KNN ",scipy.stats.ttest_rel(LR_statsScore, KNN_statsScore))
print("LR vs SVM ",scipy.stats.ttest_rel(LR_statsScore, SVM_statsScore))
print("LR vs BR ",scipy.stats.ttest_rel(LR_statsScore, BR_statsScore))
print("LR vs DR",scipy.stats.ttest_rel(LR_statsScore, DR_statsScore))


print("\n######## THE END OF PROGRAM ########")
