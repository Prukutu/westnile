import pickle

import pandas
import numpy as np

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, Binarizer

import matplotlib.pyplot as plt

df = pickle.load(open('data.p', 'rb'))

def plot_roc(ytrue, ypred, out=None):
    fpr, tpr, thresh = metrics.roc_curve(ytrue, ypred[:, 1])
    fig, ax = plt.subplots(figsize=(5, 5))


    print 'auc ', metrics.roc_auc_score(ytrue, ypred[:, 1])

    reclass = Binarizer(threshold=.10).transform(ypred)
    print 'acc ', metrics.accuracy_score(ytrue, reclass[:, 1])
    print 'prec ', metrics.precision_score(ytrue, reclass[:, 1])
    print metrics.confusion_matrix(ytrue, reclass[:, 1])


    ax.plot(fpr, tpr, color='#009688')
    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1),
        color='gray', linestyle='--')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    fig.savefig(out + '-roc.png', bbox_inches='tight')

    fpr, tpr, thresh = metrics.roc_curve(ytrue, reclass[:, 1])
    ax.plot(fpr, tpr, color='#009688')
    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1),
        color='gray', linestyle='--')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    fig.savefig(out + '-roc_reclass.png', bbox_inches='tight')

    return None


# Drop features one by one so we can test different combinations
X = df.drop(columns='WnvPresent')

X = X.drop(columns='NumMosquitos')
X = X.drop(columns='Trap')
X = X.drop(columns='Tmin')
X = X.drop(columns='Tmax')
X = X.drop(columns='Tavg')
# X = X.drop(columns='Tavg-RollingMean')
X = X.drop(columns='WetBulb')
# X = X.drop(columns='WetBulb-RollingMean')
X = X.drop(columns='WetBulb-depression')
# X = X.drop(columns='WetBulb-depression-RollingMean')
X = X.drop(columns='Cool')
X = X.drop(columns='Heat')
X = X.drop(columns='Sunrise')
X = X.drop(columns='Sunset')
X = X.drop(columns='Depart')
X = X.drop(columns='CodeSum')
X = X.drop(columns='Depth')
X = X.drop(columns='Water1')
X = X.drop(columns='SnowFall')
X = X.drop(columns='SeaLevel')
X = X.drop(columns='Station')
X = X.drop(columns='DewDiff')
X = X.drop(columns='DewPoint')
# X = X.drop(columns='DewPoint-RollingMean')
X = X.drop(columns='PrecipCum')
X = X.drop(columns='PrecipLag')
# X = X.drop(columns='PrecipCum-lag')
X = X.drop(columns='PrecipTotal')
X = X.drop(columns='ResultDir')
# X = X.drop(columns='ResultSpeed')
X = X.drop(columns='AvgSpeed')
X = X.drop(columns='StnPressure')
X = X.drop(columns='CULEX TARSALIS')
X = X.drop(columns='CULEX ERRATICUS')
X = X.drop(columns='CULEX SALINARIUS')
X = X.drop(columns='CULEX TERRITANS')
X = X.drop(columns='CULEX PIPIENS')
X = X.drop(columns='CULEX RESTUANS')
X = X.drop(columns='CULEX PIPIENS/RESTUANS')
# X = X.drop(columns='month')
X = X.drop(columns='year')
X = X.drop(columns='vector')
# X = X.drop(columns='vector_weighted')


Y = df['WnvPresent']
Y[Y > 1] = 1
print X.columns

# Whatf we cap avg temps to 28C
# tempvar = 'Tavg-RollingMean'
# X[tempvar][X[tempvar] > 26] = 26

# param_grid = {'C': [.01, .1, 1, 5],
#               'gamma': [1e-7, 1e-6, 0.00001, 0.0001]}
# clf = GridSearchCV(logitstic_estimator, params)
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y,
#                                                     test_size=.3,
#                                                     random_state=1)
# clf.fit(X, Y)
# print clf.best_params_

optimized_logistic = LogisticRegression(C=.1, penalty='l2')
sgdc_log = SGDClassifier(loss='log')
svc = svm.SVC(C=0.1, probability=True, gamma=1e-7)

X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=.2,
                                                    random_state=5,
                                                    stratify=Y)
scaler = StandardScaler()

# clf = GridSearchCV(svc, param_grid)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
#
# clf.fit(X, Y)
# print clf.best_params_


optimized_logistic.fit(X_train, y_train)
y_pred = optimized_logistic.predict(X_test)
y_probs = optimized_logistic.predict_proba(X_test)

plot_roc(y_test, y_probs, out='logistic')

# svc.fit(X_train, y_train)
# y_pred = svc.predict(X_test)
# y_probs = svc.predict_proba(X_test)
#
# plot_roc(y_test, y_probs, out='svm')
#
# sgdc_log.fit(X_train, y_train)
# y_pred = sgdc_log.predict(X_test)
# y_probs = sgdc_log.predict_proba(X_test)
#
# plot_roc(y_test, y_probs, out='sgdc')


################################################################################

# Train model on
