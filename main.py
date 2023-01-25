from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
#import torch 
import streamlit as st
import sentencepiece


def translate_text_ru_en(text):
# функция переводит вводимый текст с русского на английский
    mname = 'Helsinki-NLP/opus-mt-ru-en'
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    transl_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return transl_text

def translate_text_en_ru(text):
# функция переводит вводимый текст с английского на русский
    mname = 'Helsinki-NLP/opus-mt-en-ru'
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    transl_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return transl_text

def load_text(opt):
    if opt == 'From English to Russian':
        text = st.text_input("Enter your text in English")       
    else:
        text = st.text_input("Введите свой текст на русском")   
    return text   
    
    
st.title('Перевод / Translation') # вывод шапки
option = st.selectbox(
    'Выберите язык',
    ('С русского на английский', 'From English to Russian'))

text = load_text(option) # загрузка текста
result = st.button('Перевести') # присвоение статуса по нажатию кнопки

if result:
    if option == 'From English to Russian':
        ttext = translate_text_en_ru(text)        
    else:
        ttext = translate_text_ru_en(text)        
    st.write(ttext)      
