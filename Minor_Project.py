import numpy as np
import pandas as pd

# The file is in Latin-1
df = pd.read_csv("spam.csv", encoding="latin-1")
df.sample(5)

df.describe()

df.shape

# 1.Data Cleaning

df.info()

# Drop of unnamed columns
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)

# Rename the v1 and v2
df.rename(columns={"v1": "check", "v2": "email"}, inplace=True)
df.sample(8)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df["check"] = encoder.fit_transform(df["check"])

# Check and remove duplicate values
df.duplicated().sum()
df.drop_duplicates(keep="first")


# 2.EDA

df["check"].value_counts()

# To show the pie chart of ham and spam
import matplotlib.pyplot as plt

plt.pie(df["check"].value_counts(), labels=["ham", "spam"], autopct="%0.2f")
plt.show()

# Data is imbalanced
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

# For making a column of number of characters in a email
df["num_characters"] = df["email"].apply(len)
df.head()

# For making a column of number of words in a email
df["num_words"] = df["email"].apply(lambda x: len(nltk.word_tokenize(str(x))))
df.head()

# For making a column of number of sentences in a email
df["num_sentences"] = df["email"].apply(lambda x: len(nltk.sent_tokenize(str(x))))
df.head()

# For ham
df[df["check"] == 0][["num_characters", "num_words", "num_sentences"]].describe()

# For spam
df[df["check"] == 1][["num_characters", "num_words", "num_sentences"]].describe()

df[df["check"] == 0].count()
df[df["check"] == 1].count()


# 3.Text Preprocessing
import nltk
from nltk.corpus import stopwords

# Download the stopwords dataset
nltk.download("stopwords")
stopwords.words("english")

import string

string.punctuation


def transform_text(check):
    # Lower case
    text = check.lower()

    # Tokenizer
    text = nltk.word_tokenize(check)

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing stop words and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
ps.stem("loving")

df["transformed_text"] = df["email"].apply(transform_text)
df.head()

# WordCloud for spam
from wordcloud import WordCloud

wc = WordCloud(width=500, height=500, background_color="white")
spam_wc = wc.generate(df[df["check"] == 1]["transformed_text"].str.cat(sep=" "))
plt.imshow(spam_wc)

# WordCloud for ham
wc = WordCloud(width=500, height=500, background_color="white")
ham_wc = wc.generate(df[df["check"] == 0]["transformed_text"].str.cat(sep=" "))
plt.imshow(ham_wc)

# Find top n words in spam
spam_corpus = []
for msg in df[df["check"] == 1]["transformed_text"].tolist():
    for word in msg.split():
        spam_corpus.append(word)

from collections import Counter
import seaborn as sns

sns.barplot(
    x=pd.DataFrame(Counter(spam_corpus).most_common(50))[0],
    y=pd.DataFrame(Counter(spam_corpus).most_common(50))[1],
)
plt.xticks(rotation="vertical")
plt.show()

# Find top n words in ham
ham_corpus = []
for msg in df[df["check"] == 0]["transformed_text"].tolist():
    for word in msg.split():
        ham_corpus.append(word)

sns.barplot(
    x=pd.DataFrame(Counter(ham_corpus).most_common(50))[0],
    y=pd.DataFrame(Counter(ham_corpus).most_common(50))[1],
)
plt.xticks(rotation="vertical")
plt.show()


# 4.Model Building

# Naive Bayes Model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv = CountVectorizer()
tfidf = TfidfVectorizer()

# X =cv.fit_transform(df['transformed_text']).toarray()
X = tfidf.fit_transform(df["transformed_text"]).toarray()
X.shape
y = df["check"].values

# Training and Testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))

mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred2))
print(confusion_matrix(y_test, y_pred2))
print(precision_score(y_test, y_pred2))

bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))

# 0.95695067264574
# [[957   0]
#  [ 48 110]]
# 1.0
# tfidf-->chosen

# 5.Evaluation

from sklearn.metrics import accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

svc = SVC(kernel="sigmoid", gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver="liblinear", penalty="l1")
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    "SVC": svc,
    "KN": knc,
    "NB": mnb,
    "DT": dtc,
    "LR": lrc,
    "RF": rfc,
    "AdaBoost": abc,
    "BgC": bc,
    "ETC": etc,
    "GBDT": gbdt,
    "xgb": xgb,
}


def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy, precision


# train_classifier(svc,X_train,y_train,X_test,y_test)

accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(
        clf, X_train, y_train, X_test, y_test
    )
    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame(
    {
        "Algorithm": clfs.keys(),
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
    }
).sort_values("Precision", ascending=False)
performance_df

performance_df1 = pd.melt(performance_df, id_vars="Algorithm")
performance_df1

sns.catplot(
    x="Algorithm", y="value", hue="variable", data=performance_df1, kind="bar", height=5
)
plt.ylim(0.5, 1.0)
plt.xticks(rotation="vertical")
plt.show()

# 6.Improvement
# 1. Change the max_features parameter of TfIdf
temp_df = pd.DataFrame(
    {
        "Algorithm": clfs.keys(),
        "Accuracy_max_ft_3000": accuracy_scores,
        "Precision_max_ft_3000": precision_scores,
    }
).sort_values("Precision_max_ft_3000", ascending=False)
temp_df = pd.DataFrame(
    {
        "Algorithm": clfs.keys(),
        "Accuracy_scaling": accuracy_scores,
        "Precision_scaling": precision_scores,
    }
).sort_values("Precision_scaling", ascending=False)
new_df = performance_df.merge(temp_df, on="Algorithm")
new_df_scaled = new_df.merge(temp_df, on="Algorithm")
temp_df = pd.DataFrame(
    {
        "Algorithm": clfs.keys(),
        "Accuracy_num_chars": accuracy_scores,
        "Precision_num_chars": precision_scores,
    }
).sort_values("Precision_num_chars", ascending=False)
new_df_scaled.merge(temp_df, on="Algorithm")

# 2.Voting Classifier
svc = SVC(kernel="sigmoid", gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[("svm", svc), ("nb", mnb), ("et", etc)], voting="soft"
)
voting.fit(X_train, y_train)


y_pred = voting.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))

# 3.Applying stacking
estimators = [("svm", svc), ("nb", mnb), ("et", etc)]
final_estimator = RandomForestClassifier()

from sklearn.ensemble import StackingClassifier

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))

import pickle

pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(mnb, open("model.pkl", "wb"))

# 7.Website
# 8.Deploy
