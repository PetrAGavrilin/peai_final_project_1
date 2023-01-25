from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
import torch 
import streamlit as st
import sentencepiece


def translate_text_ru_en(text):
# функция переводит вводимый текст
    mname = 'Helsinki-NLP/opus-mt-ru-en'
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    transl_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return transl_text

def translate_text_en_ru(text):
# функция переводит вводимый текст
    mname = 'Helsinki-NLP/opus-mt-en-ru'
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
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
    ttext = translate_text_en_ru(text)
    st.write(ttext) 
