import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# load data
df = pd.read_csv('data/train.csv')
df.info(null_counts=True)

# fill in missing values and scaling 
df.Age.fillna(df.Age.median(), inplace=True)
df.Embarked.fillna('S', inplace=True)

stdsc = StandardScaler(copy=False, with_mean=False)
stdsc.fit_transform(df[['Fare', 'Age']])

# one-hot encoding on nominal features 'Sex' and 'Embarked'
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
df = pd.get_dummies(df)

# logistic regression, SVC, and random forest
# stratified k-fold cross-validation
predictors = df.columns.tolist()
predictors.remove('Survived')
svm_predictors = ['Pclass', 'Sex_male', 'Sex_female']

algos = [ [RandomForestClassifier(random_state=1, n_estimators=25, 
                        min_samples_split=5, min_samples_leaf=2), predictors],
        [LogisticRegression(random_state=1), predictors]] #,

#        [SVC(), svm_predictors] ]

clf1 = LogisticRegression(C=100, random_state=0)
clf2 = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=5, min_samples_leaf=2)
#clf3 = SVC(kernel='linear', C=1000, random_state=1, verbose=False)  
clf3 = KNeighborsClassifier()
clf3 = GaussianNB()

eclf = VotingClassifier([('lr', clf1), ('rf', clf2), ('knn', clf3)], voting='soft')
#eclf = VotingClassifier([('lr', clf1), ('rf', clf2)], voting='soft')

skf = StratifiedKFold(df.Survived, n_folds=15, random_state=1)
results = []
for train, test in skf:
    train_target = df.Survived.iloc[train]
    full_test_predictions = []
    for algo, columns in algos:
        algo.fit(df[columns].iloc[train,:], train_target)
        test_predictions = algo.predict(df[columns].iloc[test,:])
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0]+ full_test_predictions[1]) / 2.
    #test_predictions = full_test_predictions[0]+ full_test_predictions[1]+ full_test_predictions[2] 
    test_predictions[test_predictions <0.5] = 0 
    test_predictions[test_predictions>=0.5] = 1
    results.append(test_predictions)

results = np.concatenate(results, axis=0)

scores = cross_val_score(clf3, df[predictors], df['Survived'], cv=15)
print(scores.mean())

lr = KNeighborsClassifier(n_neighbors=4)
scores = cross_val_score(eclf, df[predictors], df['Survived'], cv=15)
#scores = cross_val_score(lr, df[predictors], df['Survived'], cv=15)
print('CV accuracy: %.3f +/- %.3f' % (scores.mean(), scores.std()))

asdf

# make submission
test = pd.read_csv('data/test.csv')
ids = test.PassengerId
test.Age.fillna(test.Age.median(), inplace=True)
test.Fare.fillna(test.Fare.median(), inplace=True)
test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
test = pd.get_dummies(test)

lr.fit(df[predictors], df['Survived'])
prediction = lr.predict(test)
final = pd.DataFrame({'PassengerId': ids, 
                      'Survived': prediction})
final.to_csv('titanic_submission.csv', index=False)

