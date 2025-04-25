import random
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import nlpaug.augmenter.word as naw
import nltk
from transformers import get_linear_schedule_with_warmup, DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#nltk.download('averaged_perceptron_tagger_eng')

# generates 'x' amout of samples 
def gen_samples(x):
    # num --> keyword
    kW = {1:'logic', 2:'metaphysics', 3:'epistemology', 4:'ethics', 5:'aesthetics', 6:'mind', 7:'language'}
    # keyword --> num
    wK = {'logic': 1,'metaphysics': 2,'epistemology': 3,'ethics': 4,'aesthetics': 5,'mind': 6,'language': 7}

    # number of returned papers for each keywork k
    returnV = {'logic':237, 'metaphysics':232, 'epistemology':187, 'ethics':197, 'aesthetics':26, 'mind':370, 'language':274}

    # dict of visited papers to ensure no duplicates (does not check search content)
    dupes = {'logic': [],'metaphysics': [],'epistemology': [],'ethics': [],'aesthetics': [],'mind': [],'language': []}

    samples = []
    for i in range(x):
        key = random.randint(1,7)
        word = kW[key]
        art = random.randint(1,returnV[word])
        
        # check if article has alr been visited
        if art not in dupes[word]:
            dupes[word] = dupes[word]+[art]
            samples += [f'https://quod.lib.umich.edu/p/phimp/?rgn=full+text;size=1;sort=occur;start={art};subview=short;type=simple;view=reslist;q1={word}']
    
    return samples

# checks for duplicates during data entry
def smart_sample(x):
    # load + format data
    collected = pd.read_csv("C:/Users/James/Desktop/NLP AP/Temp AP data - Sheet1.csv")
    collected.columns = ['title', 'name', 'body', 'tag', 'rating1', 'rating2', 'rating3', 'rating4']
    # remove dupes
    collected = collected[~collected['title'].str.lower().duplicated(keep=False)].reset_index(drop=True)
    # begin verifying new samples
    titles = set(collected['title'].str.lower())
    samp = 0
    while samp < x:
        # print shape for debugging
        print(f"Current shape: {collected.shape}")
        temp = input("Enter title: ").strip()

        # exit early in case you dont want to continue lol
        if temp.upper() == "I CANT FUCKING DO THIS SHIT ANYMORE":
            break

        # do nothing if False input
        if temp == "":
            print("Please fill in a valid title")
            continue
        
        # do not add duplicates
        if temp.lower() in titles:
            print("Title already exists, try again.")
            continue
        
        # Append new row
        tempA = input("Enter author name: ").strip()
        tempB = input("Enter full body: ").strip()
        # create new row
        new_row = {
            'title': temp,
            'name': tempA,
            'body': tempB,
            'tag': None,
            'rating1': None,
            'rating2': None,
            'rating3': None,
            'rating4': None
        }
        # add new row to dataframe
        collected = pd.concat([collected, pd.DataFrame([new_row])], ignore_index=True)
        titles.add(temp.lower())
        samp += 1
        print("Added to temp save.")
    # Save cleaned dataset
    return collected.to_csv('Temp AP data - Sheet1.csv', index=False)

def test():
    # test load data
    data = pd.read_csv("C:/Users/James/Desktop/NLP AP/data/ap_data.csv")
    data.columns = ['document_id', 'abstract', 'label']
    
    # test filter
    #data = data[~data['abstract'].str.lower().duplicated(keep=False)].reset_index(drop=True)
    
    # test data augmentation
    #aug = naw.SynonymAug(aug_src='wordnet')
    aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
    augmented = aug.augment(data.iloc[0,1], n=1) # change to multiply data by n-1
    new_row = {
        'document_id': 1,
        'abstract': augmented,
        'label': data.iloc[0,2],
    }
    dat = pd.DataFrame([new_row])
    print(data.iloc[0,1])
    print('============================================')
    print(dat.head())

    

def mutate():
    # read data
    data = pd.read_csv("C:/Users/James/Desktop/NLP AP/data/ap_data.csv")
    data.columns = ['document_id', 'abstract', 'label']
    # remove dupes
    data = data[~data['abstract'].str.lower().duplicated(keep=False)].reset_index(drop=True)
    ind = data.shape[0]
    start = data.shape[0]
    print(f'Intial shape {data.shape}')
    # bootstrap
    method = 'contextual' # edit for contextual
    if method == 'synonym':
        aug = naw.SynonymAug(aug_src='wordnet')
    elif method == 'contextual':
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
    else:
        raise ValueError("Unsupported method")
    # bootstrap samples
    print(f'Bootstrapping using method {method}')
    for text in data['abstract']:
        augmented = aug.augment(text, n=1) # change to multiply data by n-1
        ind += 1
        new_row = {
            'document_id': ind,
            'abstract': augmented,
            'label': data.iloc[ind-start,2],
        }
        data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)
        print(f'Bootstrapped sample {ind}!')
    print('Compiled mutated dataset!')
    print(data.shape)
    return data.to_csv('aug_ap_data.csv', index=False)
    
def train_test_split():
    # load data
    data = pd.read_csv("C:/Users/James/Desktop/NLP AP/data/ap_data.csv")
    data.columns = ['document_id', 'abstract', 'label']
    # randomize
    data = shuffle(data, random_state=42)
    # generate indexes
    total_len = len(data)
    #train_end = int(0.6 * total_len) 
    dev_end = int(0.8 * total_len)  
    # split
    train_df = data.iloc[:dev_end].reset_index(drop=True)
    dev_df = data.iloc[dev_end:].reset_index(drop=True)
    #test_df = data.iloc[dev_end:].reset_index(drop=True)
    # export
    train_df.to_csv("train.txt", sep="\t", index=False, header=True)
    dev_df.to_csv("dev.txt", sep="\t", index=False, header=True)
    #test_df.to_csv("test.txt", sep="\t", index=False, header=True)
    return

def gen_larger_test():
    # load data
    data = pd.read_csv("C:/Users/James/Desktop/NLP AP/data/ap_data.csv")
    data.columns = ['document_id', 'abstract', 'label']
    # randomize
    data = shuffle(data, random_state=42)
    # generate indexes
    total_len = len(data)
    end = int(0.5 * total_len)  
    # split
    df = data.iloc[:end].reset_index(drop=True)
    # export
    df.to_csv("test.txt", sep="\t", index=False, header=True)

def calculate_cosine_similarities():
    df = pd.read_csv("C:/Users/James/Desktop/NLP AP/data/aug_ap_data.csv")
    df.columns = ['document_id', 'abstract', 'label']
    # Determine the column containing text
    if 1 in df.columns:
        text_column = df[1]
    elif 'abstract' in df.columns:
        text_column = df['abstract']
    else:
        raise ValueError("DataFrame must contain either column index 1 or 'abstract'")
    
    # Split into original and bootstrapped samples
    original_texts = text_column.iloc[:273].tolist()
    bootstrapped_texts = text_column.iloc[273:].tolist()

    # Vectorize texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(original_texts + bootstrapped_texts)

    # Compute cosine similarities: bootstrapped vs original
    original_vectors = tfidf_matrix[:273]
    bootstrapped_vectors = tfidf_matrix[273:]
    similarities = cosine_similarity(bootstrapped_vectors, original_vectors)

    # Optional: convert to DataFrame for easier interpretation
    similarity_df = pd.DataFrame(similarities, 
                                  columns=[f'Original_{i}' for i in range(273)],
                                  index=[f'Bootstrapped_{i}' for i in range(len(bootstrapped_texts))])
    print(f'Cos similarity between original and boostrapped: {similarity_df.iloc[:,0].mean()}')
    return 

# use smart sample to add x amount of rows to the dataset
# smart_sample(1)

#test()
#mutate()
#train_test_split()
#gen_larger_test()
calculate_cosine_similarities()
