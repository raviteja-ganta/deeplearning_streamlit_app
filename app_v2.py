import numpy as np
import pandas as pd
import streamlit as st
#from annotated_text import annotated_text
#from st_annotated_text import annotated_text

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from itertools import cycle, islice

import transformers
from transformers import DistilBertForSequenceClassification, BertTokenizer, BertConfig, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup, DistilBertTokenizer 

#from seqeval.metrics import f1_score, accuracy_score

from collections import defaultdict
import base64

from pathlib import Path
import requests
import os
import urllib.request

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

main_bg = "background.jpg"
main_bg_ext = "jpg"

side_bg = "background.jpg"
side_bg_ext = "jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


pre_model_name_ner = 'bert-base-cased'
pre_model_name_sent = 'distilbert-base-cased'

tag_map_index = {0: 'PAD',
 1: 'O',
 2: 'B-geo',
 3: 'B-gpe',
 4: 'B-per',
 5: 'I-geo',
 6: 'B-org',
 7: 'I-org',
 8: 'B-tim',
 9: 'B-art',
 10: 'I-art',
 11: 'I-per',
 12: 'I-gpe',
 13: 'I-tim',
 14: 'B-nat',
 15: 'B-eve',
 16: 'I-eve',
 17: 'I-nat',
 18: '[CLS]',
 19: '[SEP]'}
 
url_sent = https://zenodo.org/api/files/c3f107db-d31e-4b27-87c0-a316620a7d3b/best_model_state_sentiment_a2.bin

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True)
    
@st.cache
def load_model_ner():
    """Loading already saved model"""
    Bertmodel = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=20, output_attentions = False, output_hidden_states = False)

    ner_model = Bertmodel
    
    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("best_model_state_ner_f2.bin")
    
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from utils import download_file_from_google_drive
            download_file_from_google_drive('1ibrN3PF45pWTPCSrCZecTUK3AzcDdrv5', f_checkpoint)
            
    
    ner_model.load_state_dict(torch.load(f_checkpoint, map_location = torch.device('cpu')))
    ner_model = ner_model.to(device)
    ner_model.eval()
    return ner_model
    



@st.cache
def load_model_sentiment():
    """Loading already saved model"""
    
    Bertmodel = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels = 2)

    sent_model = Bertmodel
    
    #save_dest = Path('model')
    #save_dest.mkdir(exist_ok=True)
    
    #f_checkpoint = Path("best_model_state_sentiment_a2.bin")
    
    #if not f_checkpoint.exists():
    #    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
    #        from utils import download_file_from_google_drive
    #        download_file_from_google_drive('1ibrN3PF45pWTPCSrCZecTUK3AzcDdrv5', f_checkpoint)
            
    
    #sent_model.load_state_dict(torch.load(f_checkpoint, map_location = torch.device('cpu')))
    sent_model.load_state_dict(torch.load(url_sent, map_location = torch.device('cpu')))
    sent_model = sent_model.to(device)
    sent_model.eval()
    return sent_model
    
@st.cache
def load_tokenizer_ner(model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer
    
@st.cache
def load_tokenizer_sent(model_name):
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    return tokenizer

##################################################################################################
def make_predictions_ner(text, inp_text,model,model_name):
    tokenizer = load_tokenizer_ner(model_name)
    tokenized_sentence = tokenizer.encode(text)
    input_id = torch.tensor([tokenized_sentence]).to(device)
    
    with torch.no_grad():
        output = model(input_id)
        
    labels = np.argmax(output[0].to('cpu').numpy(),axis = 2)
    tokens = tokenizer.convert_ids_to_tokens(input_id.to('cpu').numpy()[0])
    
    new_tokens, new_labels = [], []

    for token, label in zip(tokens,labels[0]):
        if token.startswith('##'):
            new_tokens[-1] =  new_tokens[-1] + token[2:]
  
        else:
            new_labels.append(tag_map_index.get(label))
            new_tokens.append(token)
            
    color = list(islice(cycle(['#8ef','#faa','#afa','#fea','#8ef','#afa']),len(new_tokens)))
    sent = []
    for token, label, color in zip(new_tokens, new_labels, color):
        if token == '[CLS]' or token == '[SEP]':
            continue
        if label == 'O':
            label = ' '
            sent.append(token.lower() + ' ')

        else:
            sent.append((token.upper(), label, color))
        
    return sent
##################################################################################################
def make_predictions_sent(text,model,model_name):
    class_names = ['Negative', 'Positive']
    tokenizer = load_tokenizer_sent(model_name)
    
    tokenized_sentence = tokenizer.encode(text)

    input_id = torch.tensor([tokenized_sentence]).to(device)

    with torch.no_grad():
      output = model(input_id)

    prediction = np.argmax(output[0].detach().cpu().numpy()[0], axis = 0)
    
    return class_names[prediction]
##################################################################################################	

def main():

    st.title('Deep Learning one at a time')
    
    activities = ['NER Checker','Sentiment Classification']
    choice = st.sidebar.selectbox("Select Activity",activities)
    
    #if choice == 'NER Checker':
    #	st.header('Named Entity Recognition')
    #	st.text("")
    #	st.text("")
    #	raw_text = st.text_area("Enter Text Here","Type Here")
    #	raw_text1 = raw_text.title()
    #	model = load_model_ner('Named Entity Recognition/best_model_state_ner_f2.bin')
        #model = load_model_ner('https://github.com/raviteja-ganta/deeplearning_streamlit_app/blob/main/Named%20Entity%20Recognition/best_model_state_ner_f2.bin')
    #	if st.button("Analyze"):
    #		pred = make_predictions_ner(raw_text1,raw_text,model,pre_model_name_ner)

    #		annotated_text(*pred)

    
    
    
    #	st.text('''
    #		   geo = Geographical Entity
    #		   org = Organization
    #		   per = Person
    #		   gpe = Geopolitical Entity
    #		   tim = Time indicator
    #		   art = Artifact
    #		   eve = Event
    #		   nat = Natural Phenomenon
    #		   
    #		   B - indicates first token in multi-token entity and I - indicates one in middle of multi-token entity'''
    #		   )
              
        
    if choice == 'Sentiment Classification':
        st.header('Sentiment Classification')
        st.text("")
        st.text("")
        raw_text = st.text_area("Enter Text Here","Type Here")
        
        #model = load_model_sentiment('Sentiment Analysis/best_model_state_sentiment_a2.bin')
        Bertmodel = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels = 2)

        model = Bertmodel
        
        model.load_state_dict(torch.load(url_sent, map_location = torch.device('cpu')))
        model = model.to(device)
        model.eval()
        
        #model = load_model_sentiment()
        if st.button("Analyze"):
            pred = make_predictions_sent(raw_text,model,pre_model_name_sent)
            st.text("")
            st.text("")
            if pred == 'Positive':
                st.markdown("<h1 style='text-align: center;color: green;'>Positive Sentiment</h1>", unsafe_allow_html=True)
                st.text("")
                st.markdown("<h1 style='text-align: center;color: red;'>ðŸ˜€</h1>", unsafe_allow_html=True)
            if pred == 'Negative':
                st.markdown("<h1 style='text-align: center;color: red;'>Negative Sentiment</h1>", unsafe_allow_html=True)
                st.text("")
                st.markdown("<h1 style='text-align: center;color: red;'>ðŸ˜¢</h1>", unsafe_allow_html=True)



if __name__ == '__main__':
    main()
    
    
    
    