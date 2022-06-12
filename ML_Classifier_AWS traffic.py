import pandas as pd
import numpy as np

dataset1 = pd.read_csv('Summary_datset_AWSCloud.csv')

X1 = dataset1[[
             "src_port","dst_port","bidirectional_last_seen_ms",
             "ip_version","bidirectional_duration_ms","bidirectional_packets",
             "bidirectional_bytes","src2dst_duration_ms","src2dst_packets","src2dst_bytes","dst2src_duration_ms","dst2src_packets",
             "dst2src_bytes","bidirectional_min_ps","bidirectional_mean_ps","bidirectional_stddev_ps",
             "bidirectional_max_ps","src2dst_min_ps","src2dst_mean_ps",	"src2dst_stddev_ps",
             "src2dst_max_ps","dst2src_min_ps",	"dst2src_mean_ps","dst2src_stddev_ps","dst2src_max_ps",
             "bidirectional_min_piat_ms","bidirectional_mean_piat_ms","bidirectional_stddev_piat_ms",
             "bidirectional_max_piat_ms","src2dst_min_piat_ms",	"src2dst_mean_piat_ms",
             "src2dst_stddev_piat_ms","src2dst_max_piat_ms","dst2src_min_piat_ms","dst2src_mean_piat_ms",
             "dst2src_stddev_piat_ms","dst2src_max_piat_ms","bidirectional_syn_packets","bidirectional_cwr_packets",
             "bidirectional_ece_packets","bidirectional_ack_packets",
             "bidirectional_psh_packets","bidirectional_rst_packets","bidirectional_fin_packets",
             "src2dst_syn_packets",	"src2dst_cwr_packets","src2dst_ece_packets",
             "src2dst_ack_packets",	"src2dst_psh_packets",	"src2dst_rst_packets",
             "src2dst_fin_packets",	"dst2src_syn_packets","dst2src_ece_packets",
             "dst2src_ack_packets", "dst2src_psh_packets","dst2src_rst_packets",
             "dst2src_fin_packets","src2dst_first_seen_ms","src2dst_last_seen_ms","dst2src_first_seen_ms",
             "dst2src_last_seen_ms"]]

Y1 = dataset1[["application"]]
Y1 = Y1.to_numpy()
Y1 = Y1.ravel()
labels1, uniques1 = pd.factorize(Y1)
Y1 = labels1
Y1 = Y1.ravel()
import scipy.stats as stats
X1 = stats.zscore(X1)
X1 = np.nan_to_num(X1)



dataset2 = pd.read_csv('Summary_datset_Laptop.csv')

X2 = dataset2[[
             "src_port","dst_port","bidirectional_last_seen_ms",
             "ip_version","bidirectional_duration_ms","bidirectional_packets",
             "bidirectional_bytes","src2dst_duration_ms","src2dst_packets","src2dst_bytes","dst2src_duration_ms","dst2src_packets",
             "dst2src_bytes","bidirectional_min_ps","bidirectional_mean_ps","bidirectional_stddev_ps",
             "bidirectional_max_ps","src2dst_min_ps","src2dst_mean_ps",	"src2dst_stddev_ps",
             "src2dst_max_ps","dst2src_min_ps",	"dst2src_mean_ps","dst2src_stddev_ps","dst2src_max_ps",
             "bidirectional_min_piat_ms","bidirectional_mean_piat_ms","bidirectional_stddev_piat_ms",
             "bidirectional_max_piat_ms","src2dst_min_piat_ms",	"src2dst_mean_piat_ms",
             "src2dst_stddev_piat_ms","src2dst_max_piat_ms","dst2src_min_piat_ms","dst2src_mean_piat_ms",
             "dst2src_stddev_piat_ms","dst2src_max_piat_ms","bidirectional_syn_packets","bidirectional_cwr_packets",
             "bidirectional_ece_packets","bidirectional_ack_packets",
             "bidirectional_psh_packets","bidirectional_rst_packets","bidirectional_fin_packets",
             "src2dst_syn_packets",	"src2dst_cwr_packets","src2dst_ece_packets",
             "src2dst_ack_packets",	"src2dst_psh_packets",	"src2dst_rst_packets",
             "src2dst_fin_packets",	"dst2src_syn_packets","dst2src_ece_packets",
             "dst2src_ack_packets", "dst2src_psh_packets","dst2src_rst_packets",
             "dst2src_fin_packets","src2dst_first_seen_ms","src2dst_last_seen_ms","dst2src_first_seen_ms",
             "dst2src_last_seen_ms"]]

Y2 = dataset2[["application"]]
Y2 = Y2.to_numpy()
Y2 = Y2.ravel()
labels2, uniques2 = pd.factorize(Y2)
Y2 = labels2
Y2 = Y2.ravel()

X2 = stats.zscore(X2)
X2 = np.nan_to_num(X2)

from sklearn.model_selection import train_test_split
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, random_state=0, test_size=0.5)
X_test2 = X2
Y_test2 = Y2


from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, SGD
import tensorflow as tf



Ncol = X_train1.shape[1];  # Number of features
nOut = len(np.unique(Y_train1))  # Number of classes

model = Sequential()
model.add(Dense(1000, input_dim=Ncol, activation='relu'))
model.add(Dense(550, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(125, activation='relu'))
model.add(Dense(nOut, activation='softmax'))
rmsprop = RMSprop(lr=0.001)
sgd = SGD(lr=0.1)
# adam
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['sparse_categorical_accuracy'])
#model.fit(X_train1, Y_train1, validation_data=(X_test1, Y_test1), epochs=100, batch_size=30, verbose=1)
model.fit(X_train1, Y_train1, epochs=100, batch_size=30, verbose=1)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score, multilabel_confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
#clf = DecisionTreeClassifier(random_state=0, max_depth=6)
#clf = RandomForestClassifier(random_state=0,max_depth=8)


cv = KFold(n_splits=10, random_state=0, shuffle=True)
predict = np.argmax(model.predict(X_test1),axis=1)
accuracy = accuracy_score(Y_test1, predict)
precision = precision_score(Y_test1, predict, average='weighted')
recall = recall_score(Y_test1, predict, average='weighted')
f1scoreMacro = f1_score(Y_test1, predict, average='macro')
cm_Avids = confusion_matrix(Y_test1, predict)


import seaborn as sn
import matplotlib.pyplot as plt

labels = uniques1
plt.figure(2,figsize=(5, 2))
plt.title("Confusion Matrix", fontsize=10)
# Normalise
cmnorm = cm_Avids.astype('float') / cm_Avids.sum(axis=1)[:, np.newaxis]
sn.heatmap(cmnorm, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=labels, yticklabels=labels)


labels = uniques1
plt.figure(3,figsize=(5, 2))
plt.title("Confusion Matrix", fontsize=10)
sn.heatmap(cm_Avids, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=labels, yticklabels=labels)



# Facebook
TP = cm_Avids [0, 0];
TN = cm_Avids [1, 1] + cm_Avids [2, 2] + cm_Avids [1, 2] + cm_Avids [2, 1];
FP = cm_Avids [1, 0] + cm_Avids [2, 0];
FN = cm_Avids [0, 1] + cm_Avids [0, 2];

# Twitch
TP = cm_Avids [1, 1];
TN = cm_Avids [0, 0] + cm_Avids [2, 2] + cm_Avids [0, 2] + cm_Avids [2, 0];
FP = cm_Avids [0, 1] + cm_Avids [2, 1];
FN = cm_Avids [1, 0] + cm_Avids [1, 2];

# Youtube
TP = cm_Avids [2, 2];
TN = cm_Avids [0, 0] + cm_Avids [1, 1] + cm_Avids [1, 0] + cm_Avids [0, 1];
FP = cm_Avids [1, 2] + cm_Avids [0, 2];
FN = cm_Avids [2, 0] + cm_Avids [2, 1];






FP = cm_Avids.sum(axis=0) - np.diag(cm_Avids)  
FN = cm_Avids.sum(axis=1) - np.diag(cm_Avids)
TP = np.diag(cm_Avids)
TN = cm_Avids.sum() - (FP + FN + TP)

Accuracy = (TP+TN)/(TP+TN+FP+FN);
Precision = TP/(TP+FP);
Recall = TP/(TP+FN);
F_measure = (2*Recall*Precision)/(Recall+Precision);


# Local PC

predict_2 = np.argmax(model.predict(X2),axis=1)
accuracy_2 = accuracy_score(Y2, predict_2)
precision_2 = precision_score(Y2, predict_2, average='weighted')
recall_2 = recall_score(Y2, predict_2, average='weighted')
f1scoreMacro_2 = f1_score(Y2, predict_2, average='macro')
cm_local = confusion_matrix(Y2, np.argmax(model.predict(X2),axis=1))

mcm = multilabel_confusion_matrix(Y2, np.argmax(model.predict(X2),axis=1))

tn = mcm[:, 0, 0]
tp = mcm[:, 1, 1]
fn = mcm[:, 1, 0]
fp = mcm[:, 0, 1]


Accuracy_T = (tp+tn)/(tp+tn+fp+fn);
Precision_T = tp/(tp+fp);
Recall_T = tp/(tp+fn);
F_measure_T = (2*Recall_T*Precision_T)/(Recall_T+Precision_T);



print(tp, tn, fp, fn, Accuracy_T, Precision_T, Recall_T, F_measure_T)
print(TP, TN, FP, FN, Accuracy, Precision, Recall, F_measure)
print(accuracy, precision, recall, f1scoreMacro)
print(accuracy_2, precision_2, recall_2, f1scoreMacro_2)


labels = uniques1
plt.figure(4,figsize=(5, 2))
plt.title("Confusion Matrix", fontsize=10)
sn.heatmap(cm_local, annot=True, cmap="YlGnBu", fmt=".2f", xticklabels=labels, yticklabels=labels)

# from statistics import mean

# mean(Accuracy_T)

# mean(Precision_T)

# mean(Recall_T)

# mean(F_measure_T)









