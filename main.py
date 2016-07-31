import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# load data
df = pd.read_csv('train.csv')
df.info(null_counts=True)

# fill in missing values and scaling 
#df.loc[df.Fare <5, 'Fare'] = df.Fare.median()
df.Age.fillna(df.Age.median(), inplace=True)
df.Embarked.fillna('S', inplace=True)
#stdsc = StandardScaler(copy=False, with_mean=False)
#stdsc.fit_transform(df[['Fare', 'Age']])

# split Age by 16, i.e., child = 0 and adult = 1
df.loc[df.Age<16,  'Age'] = 0
df.loc[df.Age>=16, 'Age'] = 1

# one-hot encoding on nominal features 'Sex' and 'Embarked'
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
df = pd.get_dummies(df)

# logistic regression, SVC, and random forest
# stratified k-fold cross-validation
predictors = df.columns.tolist()
predictors.remove('Survived')
#svm_predictors = ['Pclass', 'Sex_male', 'Sex_female']

clf1 = LogisticRegression(C=1, random_state=1, warm_start=True, penalty='l2')
clf2 = RandomForestClassifier(random_state=1, n_estimators=10, 
                min_samples_split=5, min_samples_leaf=2, max_features=3)
eclf = VotingClassifier([('lr', clf1), ('rf', clf2)], voting='soft')

scores = cross_val_score(clf1, df[predictors], df['Survived'], cv=15)
print(scores.mean())
scores = cross_val_score(clf2, df[predictors], df['Survived'], cv=15)
print(scores.mean())

scores = cross_val_score(eclf, df[predictors], df['Survived'], cv=15)
print('CV accuracy: %.3f +/- %.3f' % (scores.mean(), scores.std()))

adsfa

# make submission
test = pd.read_csv('test.csv')
ids = test.PassengerId
test.Age.fillna(test.Age.median(), inplace=True)
test.Fare.fillna(test.Fare.median(), inplace=True)
test.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
test = pd.get_dummies(test)

eclf.fit(df[predictors], df['Survived'])
prediction = eclf.predict(test)
final = pd.DataFrame({'PassengerId': ids, 
                      'Survived': prediction})
final.to_csv('titanic_submission.csv', index=False)

