# import streamlit as st
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# import nltk 
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import string
# nltk.download('punkt')
# nltk.download('stopwords')
# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     text = [word for word in text if word.isalnum()]

#     text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

#     text = [ps.stem(word) for word in text]

#     return " ".join(text)

# st.title("Spam Email Detection")
# input_sms = st.text_area("Enter your SMS")

# if st.button("Predict"):
#   transformed_sms = transform_text(input_sms)
#   vector_input = tfidf.transform([transformed_sms])
#   result = model.predict(vector_input)[0]
#   if result == 1:
#     st.header("Spam")
#   else:
#     st.header("Not Spam")


import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('popular')
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

ps = PorterStemmer()



# Load the pre-trained model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

st.title("Spam Email Detection")
input_sms = st.text_area("Enter your SMS")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    predicted_prob = model.predict_proba(vector_input)[0][1]  # Probability of being spam
    # st.write(f"Predicted Probability (Spam): {predicted_prob:.4f}")

    # Adjust the threshold (you can experiment with different values)
    threshold = 0.5
    if predicted_prob >= threshold:
        st.header("Spam")
    else:
        st.header("Not Spam")
