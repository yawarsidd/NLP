#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install tensorflow-hub


# In[3]:


import tensorflow as tf
import tensorflow_hub as hub


# In[5]:


pip install tensorflow-text


# In[6]:


import tensorflow_text as text


# In[7]:


bert_preprocess=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[8]:


bert_preprocess_model=hub.KerasLayer(bert_preprocess)


# In[9]:


text_test=["The cat chased its tail in circles, circles.","I love python programming "]
text_preprocess=bert_preprocess_model(text_test)


# In[46]:


text_preprocess['input_mask']   #12 1's means cls then 9 words and then sep for 1st sentence means token of the sentence


# In[10]:


bert_model=hub.KerasLayer(bert_encoder)
bert_results=bert_model(text_preprocess)


# In[11]:


bert_results['pooled_output']  # sentence Embeddings


# In[41]:


# Word Embeddings


# In[62]:


bert_results['sequence_output']  


# In[ ]:


#This is a matrix or tensor with shape (2, 128, 768), where 2 is the number of sentences, 
#128 is the padded length of each sentence and 768 is the size of the embedding vector for each word or token in the sentence.
#Each element of the sequence_output is a vector representing the contextualized embedding for a token in an input sentence. 
#The padding is added to all sequences to make them uniform length and allow for efficient parallel processing. 
#The padded positions are filled with zeros which do not contribute to the model’s output.


# In[14]:


pip install transformers


# In[28]:


from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Input sentence
sentence = "This is an example sentence for obtaining word embeddings using BERT."

# Tokenize input sentence
tokens = tokenizer.encode(sentence, add_special_tokens=True)

# Convert tokens to a TensorFlow tensor
input_ids = tf.constant([tokens])

# Get word embeddings
outputs = model(input_ids)

# Extract word embeddings from BERT model outputs
word_embeddings = outputs.last_hidden_state

print("Word embeddings shape:", word_embeddings.shape)


# In[29]:


word_embeddings


# In[63]:



# Input sentence
sentence = "This is an example sentence for obtaining word embeddings using BERT."

# Tokenize input sentence
tokens = tokenizer.encode(sentence, add_special_tokens=True)

# Convert tokens to a TensorFlow tensor
input_ids = tf.constant([tokens])

# Get word embeddings
outputs = model(input_ids)

# Extract word embeddings from BERT model outputs
word_embeddings = outputs.last_hidden_state

# Remove padding tokens from word embeddings
padding_mask = tf.cast(input_ids != tokenizer.pad_token_id, word_embeddings.dtype)
word_embeddings = word_embeddings * tf.expand_dims(padding_mask, axis=-1)

# Convert token IDs to words
words = tokenizer.convert_ids_to_tokens(tokens)

# Convert word embeddings to a DataFrame
df = pd.DataFrame(word_embeddings.numpy().squeeze(), columns=[f"vector_{i}" for i in range(word_embeddings.shape[-1])])
df['word'] = words

# Drop rows corresponding to padding tokens
df = df[df['word'] != '[PAD]']

# Reset index
df.reset_index(drop=True, inplace=True)

print(df)


# In[58]:


sentence = ["This is an example sentence for obtaining word embeddings using BERT."]


# In[60]:


text_preprocess=bert_preprocess_model(sentence)
text_preprocess['input_mask']   # here also 17's one indicate divide into 17 tokens and then padded zeroes to make a shape of 128


# In[ ]:


#padding is used to ensure that all sequences within a batch have the same length.
#Text=["This is an example sentence.", "Another sentence for illustration.", "A third sentence."]

#Sentence1= ["[CLS]", "This", "is", "an", "example", "sentence", ".", "[SEP]"]
#Sentence2= ["[CLS]", "Another", "sentence", "for", "illustration", ".", "[SEP]"]
#Sentence3= ["[CLS]", "A", "third", "sentence", ".", "[SEP]"]

#Sentence 1: 8 tokens
#Sentence 2: 7 tokens
#Sentence 3: 6 tokens

#The longest sequence has 8 tokens. Therefore, we pad the shorter sequences with [PAD] tokens to match this length
#After padding sentences looks like:-
#Sentence1=  ["[CLS]", "This", "is", "an", "example", "sentence", ".", "[SEP]"]
#Sentence2=  ["[CLS]", "Another", "sentence", "for", "illustration", ".", "[SEP]", "[PAD]"]
#Sentence3=  ["[CLS]", "A", "third", "sentence", ".", "[SEP]", "[PAD]", "[PAD]"]
# in general word scenario it is padded with generally 128 for each sentences


# In[40]:


# Two sentences with different word limits
sentences = [
    "This is the first sentence with the fewer words.",
    "This is the second sentence with more words and a higher word limit."
]

# Tokenize sentences and convert to input IDs
input_ids = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences]

# Get maximum length among the input sequences
max_length = max(len(ids) for ids in input_ids)

# Pad input sequences to the maximum length
input_ids_padded = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]

# Convert to TensorFlow tensors
input_ids_tensor = tf.constant(input_ids_padded)

# Get word embeddings
outputs = model(input_ids_tensor)

# Extract word embeddings from BERT model outputs for both sentences
word_embeddings = outputs.last_hidden_state.numpy()

# Print words along with their embedding vectors for both sentences
for i, embeddings in enumerate(word_embeddings):
    words = tokenizer.convert_ids_to_tokens(input_ids_padded[i])
    print(f"Sentence {i + 1}:")
    for word, embedding in zip(words, embeddings):
        print(f"Word: {word}, Embedding: {embedding}")
    print()


# In[64]:


# sentence1: word embeddings for [CLS], Each word, [Sep], then 4 pads beacause sentence 1 has less words so to make parallel with the 2nd senetnce
# sentence2: word Embeddings for {CLS}, Each word ,[SEP]


# In[65]:


# End


# In[ ]:




