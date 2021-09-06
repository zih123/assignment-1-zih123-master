# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import math
import os
import re

import nltk
nltk.download('punkt')
import numpy as np
from numpy.linalg import norm
import pandas as pd
from matplotlib import pyplot as plt
import json
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

def task1():
    #Complete task 1 here
    teams_codes = []
    with open(datafilepath,'r') as fp:
        data = json.load(fp)
        teams_codes = data['teams_codes']
        teams_codes.sort()
    return teams_codes
    
def task2():
    #Complete task 2 here

    # read the file and get corresponding information
    scores = []
    with open(datafilepath,'r') as fp:
        data = json.load(fp)
        clubs = data['clubs']
        for club in clubs:
            scores.append([club['club_code'],club['goals_scored'],club['goals_conceded']])

        scores.sort(key=lambda item:item[0])

    # store the data in the file
    with open('task2.csv','w') as fp:
        fp.write('team code, goals scored by team, goals scored against team\n')
        for tt in scores:
            fp.write('{0},{1},{2}\n'.format(tt[0],tt[1],tt[2]))
    return scores
      
def task3():
    #Complete task 3 here
    paths = os.listdir(articlespath)
    paths.sort()
    reg = re.compile(r'\d+-\d+')
    scores = []
    for path in paths:
        with open(articlespath+'/'+path, 'r') as fp:
            text = ' '.join(fp.read().splitlines())
            # match the score
            ll = reg.findall(text)
            max_score = 0
            if len(ll) == 0:
                scores.append([path,max_score])
                continue
            # find the maximum score and add it to the list
            for tt in ll:
                ss = tt.split('-')
                s1 = int(ss[0])
                s2 = int(ss[1])
                if s1 < 0 or s1 > 99 or s2 < 0 or s2 > 99:
                    continue
                value = s1 + s2
                if value > max_score:
                    max_score = value
            scores.append([path, max_score])

    # write the data to the file
    with open('task3.csv','w') as fp:
        fp.write('filename, total goals\n')
        for tt in scores:
            fp.write('{0},{1}\n'.format(tt[0], tt[1]))
    return

def task4():
    #Complete task 4 here
    x = pd.read_csv('task3.csv').values[:,1].tolist()
    plt.boxplot(x,labels=[''],whis=1.5)
    plt.title('Task4: boxplot of total scores')
    plt.xlabel(u'total_scores')
    plt.ylabel(u'values')
    plt.savefig('task4.png')
    return
    
def task5():
    #Complete task 5 here
    mentions = {}
    # read names of clubs
    with open(datafilepath) as fp:
        data = json.load(fp)
        name_clubs = data['participating_clubs']

    for tt in name_clubs:
        mentions[tt] = 0

    # traverse the files and calculate the number of mentions
    paths = os.listdir(articlespath)
    paths.sort()
    for path in paths:
        with open(articlespath + '/' + path, 'r') as fp:
            text = ' '.join(fp.read().splitlines())

            for club in name_clubs:
                if text.find(club) != -1:
                    mentions[club] += 1

    # put the data in the list
    list_mentions = []
    for key in mentions.keys():
        list_mentions.append([key, mentions[key]])

    # sort the list in ascending alphabetic order
    list_mentions.sort(key=lambda item: item[0])

    # write the data to the file
    with open('task5.csv','w') as fp:
        fp.write('club name, number of mentions\n')
        for tt in list_mentions:
            fp.write('{0},{1}\n'.format(tt[0], tt[1]))

    # plot a bar chart and save it to a image file
    x = []
    y = []
    for tt in list_mentions:
        x.append(tt[0])
        y.append(tt[1])

    plt.bar(x, y)
    plt.title('Task5: number of mentions for each club name')
    plt.xticks(rotation=-90)
    plt.xlabel(u'club name')
    plt.ylabel(u'number of mentions')
    plt.tight_layout()
    plt.savefig('task5.png')
    return
    
def task6():
    #Complete task 6 here
    # get mentioned articles of each club
    mentions = {}
    # read names of clubs
    with open(datafilepath) as fp:
        data = json.load(fp)
        name_clubs = data['participating_clubs']

    for tt in name_clubs:
        mentions[tt] = {'number_mentions':0,'articles':[]}

    # traverse the files ,calculate the number of mentions
    # and record mentioned articles of each club
    paths = os.listdir(articlespath)
    paths.sort()
    for path in paths:
        with open(articlespath + '/' + path, 'r') as fp:
            text = ' '.join(fp.read().splitlines())

            for club in name_clubs:
                if text.find(club) != -1:
                    mentions[club]['number_mentions'] += 1
                    mentions[club]['articles'].append(path)

    # calculate the similarity
    arr = np.zeros((len(name_clubs),len(name_clubs)))
    for i in range(len(name_clubs)):
        for j in range(len(name_clubs)):
            value = len(set(mentions[name_clubs[i]]['articles']).intersection(set(mentions[name_clubs[j]]['articles'])))
            sum_mentions = mentions[name_clubs[i]]['number_mentions']+mentions[name_clubs[j]]['number_mentions']
            if sum_mentions == 0:
                continue
            arr[i][j] = (2*value)/sum_mentions

    # draw the heatmap
    fig = plt.figure()
    sns.heatmap(arr,cmap="OrRd")
    plt.xlabel(u'clubs')
    plt.ylabel(u'clubs')
    plt.xticks(range(len(name_clubs)),name_clubs,rotation=-90)
    plt.yticks(range(len(name_clubs)),name_clubs,rotation=0)
    plt.title('Task6: heatmap of similarity')
    plt.tight_layout()
    fig.savefig("task6.png")
    return
    
def task7():
    #Complete task 7 here
    scores = pd.read_csv('task2.csv').values[:,1]
    number_mentions = pd.read_csv('task5.csv').values[:,1]
    plt.scatter(number_mentions,scores)
    plt.xlabel('number of mentions')
    plt.ylabel('scores of clubs')
    plt.title('Task7: scatterplot of number of mentions and scores')
    plt.savefig('task7.png')
    return
    
def task8(filename):
    #Complete task 8 here
    # read stop words
    stop_words = set()
    with open('stopwords_english','r') as fp:
        stop_words = fp.read().splitlines()
    # read all characters in the file
    with open(filename, 'r') as fp:
        text = ' '.join(fp.read().splitlines())

    # preprocess the text
    # change all uppercase characters to lower case
    filtered_text = text.lower()
    # remove all non-alphabetic characters exception for spacing characters
    filtered_text = re.sub('[^a-z]',' ',filtered_text)

    # tokenize the resulting string into words
    words = nltk.word_tokenize(filtered_text)

    # remove all stopwords and words which are a single character long
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]

    return filtered_words
    
def task9():
    #Complete task 9 here
    # get words of all files
    paths = os.listdir(articlespath)
    paths.sort()
    number_articles = len(paths)
    words_article = []
    for path in paths:
        filename = articlespath + '/' + path
        words_article.append(task8(filename))

    # calculate TF-IDF
    vectorize = CountVectorizer()
    transformer = TfidfTransformer()

    txtList = []
    for tt in words_article:
        val = ' '.join(tt)
        txtList.append(val)

    tf = vectorize.fit_transform(txtList)

    # calculate cosine similarity measure
    sim = {}
    tf_idf_array = transformer.fit_transform(tf).toarray()
    rows = tf_idf_array.shape[0]
    for i in range(rows):
        for j in range(rows):
            if i == j:
                continue
            if paths[j]+','+paths[i] in sim.keys():
                continue

            vec1 = tf_idf_array[i,:]
            vec2 = tf_idf_array[j,:]

            v1 = np.linalg.norm(vec1)
            v2 = np.linalg.norm(vec2)
            v3 = np.dot(vec1,vec2)
            value = v3/(v1*v2)
            sim[paths[i]+','+paths[j]] = value

    # get the top 10
    ll = []
    for tt in sim:
        ll.append([tt, sim[tt]])
    ll.sort(key=lambda item:item[1],reverse=True)

    sim_top10 = ll[:10]

    # store the top 10 to the file
    with open('task9.csv','w') as fp:
        fp.write('article1,article2, similarity\n')
        for tt in sim_top10:
            fp.write('{0},{1}\n'.format(tt[0],tt[1]))
    return