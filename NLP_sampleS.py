import random
import pandas as pd
import numpy as np

startInd = 216
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

# use smart sample to add x amount of rows to the dataset
smart_sample(15)
