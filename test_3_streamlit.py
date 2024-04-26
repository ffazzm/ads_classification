from matplotlib import category
import streamlit as st
from transformers import pipeline
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  
stop_words = stopwords.words('english')

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer() 

def clean_text(text):
    
    text = text.lower()
    
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
    for p in punctuations:
        text = text.replace(p,' ') #Removing punctuations
    
    # pattern = r"\b(?=[mdclxviι])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})([iι]x|[iι]v|v?[iι]{0,3})\b\.?"
    # text =  re.sub(pattern, ' ', text.lower())

    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")

    text = re.sub(r"http\S+", "",text) #Removing URLs 

    html=re.compile(r'<.*?>') 
    
    text = html.sub(r'',text) #Removing html tags
        
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    
    text = [lemmatizer.lemmatize(word) for word in text]
    
    text = " ".join(text) #removing stopwords
    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text) #Removing emojis
    return text

def main():
    st.title("Ads Classification")
    # section_text = st.text_input("Enter section", "")
    section_text =st.selectbox(
    'Select section',
    ('for-sale', 'housing', 'community', 'services'))
    header_text = st.text_input("Enter header", "")

    input_text = f"{section_text} {clean_text(header_text)}"
    
    if st.button("Analyze"):
        result = model(input_text)
        print(result)
        st.write("Prediction:", result[0]['label'], "| Score:", result[0]['score'])

model = pipeline("text-classification", model='saved_model')

if __name__ == "__main__":
    main()