from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch 
import streamlit as st


def translate_text(text):
# функция переводит вводимый текст
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ru-en")    
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    transl_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return transl_text

def load_text():
    text = st.text_input("Enter your text in English")
    return text   
    
    
st.title('Перевод с русского на английский') # вывод шапки
text = load_text() # загрузка текста
result = st.button('Перевести') # присвоение статуса по нажатию кнопки

if result:
    ttext = translate_text(text)
    st.write(ttext) 
