


#Import libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import xlrd
from sklearn.cross_validation import train_test_split


def process_text(text):
    # 1 Remove Punctuationa
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # 2 Remove Stop Words
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    # 3 Return a list of clean words
    return clean_words



#Load the data

old_df = pd.read_excel('new_data.xlsx')


#listID ,timestamp ,sender_name ,message ,site_listID ,SPAM

df=old_df[['message','SPAM']]

nltk.download('stopwords')

# print(df['message'].head().apply(process_text))

from sklearn.feature_extraction.text import CountVectorizer

messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['message'])


X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['SPAM'], test_size = 0.20, random_state = 0)

print("Message bow",messages_bow.shape)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train ,pred ))
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred))


from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))

# from sklearn.externals import joblib
#
# # Save the model as a pickle in a file
# joblib.dump(classifier, 'model_weights.pkl')
# joblib.dump(messages_bow, 'model_weights.pkl')
#
#
# while True:
#     x = []
#     inner_list = []
#     ans = input("Input Email")
#     inner_list.append(ans)
#     x.append(inner_list)
#     print("prediction",classifier.predict(x))