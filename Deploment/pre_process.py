def preprocess_text(text_data):
    import numpy as np
    import re
    import nltk
    def decontracted(phrase): # Function to Decontract words like 'won't' to 'will not' and so on
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"didn't", "did not", phrase)
        phrase = re.sub(r"havn't", "have not", phrase)
        phrase = re.sub(r"hasn't", "has not", phrase)
        phrase = re.sub(r"can't", "can not", phrase)
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    preprocessed_text = []
    for sentance in (text_data): # Function to lemmtize, Stem words and further cleaning of text
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        preprocessed_text.append(sent.lower().strip())
    return np.array(preprocessed_text)
