import streamlit as st
import re
from vocabulary import Vocabulary, iVocabulary

# Initialize the Streamlit app
st.title("Text Prediction Application")

# User input
input_text = st.text_input("Enter the input text:", "Once upon a time")
context_length = st.selectbox("Context length", [5,10,15])
embedding_dim = st.selectbox("Embedding dimension", [128, 64])
# embedding_dim = st.slider("Embedding dimension", 16, 128, 64)
activation_function = st.selectbox("Activation function", ["relu", "tanh"])
max_len = st.number_input("Maximum lenght of predicted text", min_value=0, max_value=1000, value=30)
# num_words_to_predict = st.slider("Number of words to predict", 1, 50, 10)
input_text = re.sub(r'[^a-zA-Z0-9 \.]', '', input_text)
input_text = str.lower(input_text)



import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if activation_function == 'relu':
    class NextWord(nn.Module):
        def __init__(self,block_size,vocab_size,emb_dim,hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size,emb_dim)
            self.lin1 = nn.Linear(block_size*emb_dim,hidden_size)
            self.lin2 = nn.Linear(hidden_size,vocab_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0],-1)
            x = self.relu(self.lin1(x))
            x = self.lin2(x)
            return x
    # model_128_5_relu = NextWord(5, len(Vocabulary), 128, 1024).to(device)
    # model_128_10_relu = NextWord(10, len(Vocabulary), 128, 1024).to(device)
    # model_128_15_relu = NextWord(15, len(Vocabulary), 128, 1024).to(device)

    # model_64_5_relu = NextWord(5, len(Vocabulary), 64, 1024).to(device)
    # model_64_10_relu = NextWord(10, len(Vocabulary), 64, 1024).to(device)
    # model_64_15_relu = NextWord(15, len(Vocabulary), 64, 1024).to(device)
    pred_model = NextWord(context_length, len(Vocabulary), embedding_dim, 1024).to(device)
    pred_model.load_state_dict(torch.load(f"models/model_{embedding_dim}_{context_length}_relu.pth", map_location=device))

if activation_function == 'tanh':
    class NextWord(nn.Module):
        def __init__(self,block_size,vocab_size,emb_dim,hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size,emb_dim)
            self.lin1 = nn.Linear(block_size*emb_dim,hidden_size)
            self.lin2 = nn.Linear(hidden_size,vocab_size)
            self.tanh = nn.Tanh()

        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0],-1)
            x = self.tanh(self.lin1(x))
            x = self.lin2(x)
            return x
    # model_128_5_relu = NextWord(5, len(Vocabulary), 128, 1024).to(device)
    # model_128_10_relu = NextWord(10, len(Vocabulary), 128, 1024).to(device)
    # model_128_15_relu = NextWord(15, len(Vocabulary), 128, 1024).to(device)

    # model_64_5_relu = NextWord(5, len(Vocabulary), 64, 1024).to(device)
    # model_64_10_relu = NextWord(10, len(Vocabulary), 64, 1024).to(device)
    # model_64_15_relu = NextWord(15, len(Vocabulary), 64, 1024).to(device)
    pred_model = NextWord(context_length, len(Vocabulary), embedding_dim, 1024).to(device)
    pred_model.load_state_dict(torch.load(f"models/model_{embedding_dim}_{context_length}_tanh.pth", map_location=device))

def generate_para(model, Vocabulary, iVocabulary, block_size, user_input=None, max_len=30):
    # Initialize context with user-provided input or default to [0] * block_size if None
    if user_input:
        # Tokenize user input to numerical IDs
        context = [Vocabulary.get(word, 0) for word in user_input.split()]
        # Pad or truncate context to match block_size
        context = context[-block_size:] if len(context) >= block_size else [0] * (block_size - len(context)) + context
    else:
        # Default context with all zeros
        context = [0] * block_size

    new_para = ' '.join([iVocabulary.get(idx, '') for idx in context]).strip()
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        # Sample from the predicted logits
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        word = iVocabulary[ix]
        # charec = list(word)
        # Append the word to the generated paragraph
        new_para = new_para + " " + word
        # Update context for next prediction
        context = context[1:] + [ix]
        # if '.' == word[-1]:
        #     break

    return new_para

if st.button("Generate Prediction"):
    predicted_text = generate_para(pred_model,Vocabulary,iVocabulary,context_length,input_text,max_len)
    st.write("Predicted Text:", predicted_text)
    # st.write("Predicted Text:",)