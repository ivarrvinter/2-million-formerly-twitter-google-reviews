import pandas as pd
import nltk
import re
import spacy
import keras
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

class Preprocessing:
    def __init__(self, df):
        self.df = df

    def lowercase(self, cell):
        if isinstance(cell, str):
            return cell.lower()
        return cell

    def find_nonalphanumeric(self, column_name):
        if column_name in self.df.columns:
            column_data = self.df[column_name].astype(str)
            nonalphanumeric_chars = set()
            for text in tqdm(column_data, desc='Finding Non-Alphanumeric Chars', unit='text'):
                nonalphanumeric_chars.update(re.findall(r'[^\w\s]', text))
            return nonalphanumeric_chars
        return set()

    def remove_chars_from_column(self, column_name, chars_to_remove):
        if column_name in self.df.columns:
            pattern = '|'.join(re.escape(char) for char in chars_to_remove)
            self.df[column_name] = self.df[column_name].astype(str).apply(lambda x: re.sub(pattern, '', x))
        return self.df

    def preprocess(self, column_name):
        self.df[column_name] = self.df[column_name].apply(lambda x: self.lowercase(x))
        nonalphanumeric_chars = self.find_nonalphanumeric(column_name)
        self.df = self.remove_chars_from_column(column_name, nonalphanumeric_chars)
        return self.df
      
class Embedding:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_embeddings(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu()

    def generate_bert_embeddings(self, tokenized_sentences):
        embeddings = []
        for sentence in tqdm(tokenized_sentences, desc='Generating Embeddings', unit='sentence'):
            sentence_embedding = self.get_embeddings(sentence)
            embeddings.append(sentence_embedding)
        return embeddings


df = pd.read_csv('/kaggle/input/2-million-formerly-twitter-google-reviews/TWITTER_REVIEWS.csv')
df.drop(df.columns[[0, 1, 2, 3, 6, 7, 8]], axis=1, inplace=True)
preprocessor = Preprocessing(df)
processed_df = preprocessor.preprocess('review_text')
bert_embeddings = Embedding('bert-base-uncased')
embeddings = bert_embeddings.generate_bert_embeddings(processed_df.iloc[:, 0])
df['BERT_Embeddings'] = embeddings
