from sklearn.externals import joblib
from nltk.corpus import stopwords
import string
# Load the model from the file
trained_from_joblib = joblib.load('model_weights.pkl')
messages_bow = joblib.load('model_weights.pkl')
# Use the loaded model to make predictions

def process_text(text):
    # 1 Remove Punctuationa
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # 2 Remove Stop Words
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    # 3 Return a list of clean words
    return clean_words


from sklearn.feature_extraction.text import CountVectorizer

while True:
    x=[]
    inner_list=[]
    ans=input("Input Email")
    inner_list.append(ans)
    x.append(inner_list)
    messages=CountVectorizer(analyzer=process_text).fit_transform(x)
    pred = trained_from_joblib.predict(messages)
    print(pred)