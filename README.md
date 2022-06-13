# Classification-of-mail-as-spam-or-ham-using-machine-learning.
The project classifies an email as spam or ham(not spam) using Machine learning Algorithm. This project is useful for every single person who uses email for communication.The email message is fed as input and Multinomial Naive Bayes Algorithm is used to predict the email message as spam or ham. Technology used is Jupyter notebook with Python as programming language and HTML,CSS and PHP as Front End.
# The Python Code for back end is as follows :
# importing modules.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score

df = pd.read_csv(r'C:\\Users\\Shivkesh\\Documents\\Spam_Detection\\mail_data.csv')
df.rename(columns = {'Category' : 'target','Message' : 'text'},inplace = True)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df = df.drop_duplicates(keep = 'first')

ps = PorterStemmer()

def transform_text(text) :
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text :
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text :
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text :
        y.append(ps.stem(i))

    return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)

tfidf = TfidfVectorizer(max_features = 3000)
x = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)
mnb = MultinomialNB()
mnb.fit(x_train,y_train)
y_pred = mnb.predict(x_test)
y_pred1 = mnb.predict(x_train)

print("accuracy_score on test data: ",accuracy_score(y_test,y_pred))
print("precision_score on test data: ",precision_score(y_test,y_pred))
print("accuracy_score on train data: ",accuracy_score(y_train,y_pred1))
print("precision_score on train data: ",precision_score(y_train,y_pred1))

input_mail = ''' '''
transform = transform_text(input_mail)

input_data = tfidf.transform([transform])

predict = mnb.predict(input_data)

if (predict == 1):
    print("Spam")
else:
    print("Ham")
