from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
import streamlit as st
import sentencepiece

def test_translate_text_ru_en():
# проверка перевода вводимого текста с русского на английский
    text = "меня зовут Иван"
    check_text = "my name is Ivan"
    mname = 'Helsinki-NLP/opus-mt-ru-en'
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    transl_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert transl_text == check_text

def test_translate_text_en_ru(text):
# проверка перевода вводимого текста с английского на русский
    check_text = "меня зовут Иван"
    text = "my name is Ivan"
    mname = 'Helsinki-NLP/opus-mt-en-ru'
    tokenizer = MarianTokenizer.from_pretrained(mname)
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids)
    transl_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert transl_text == check_text


    
