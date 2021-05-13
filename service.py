#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author   : MuyunLi
Date     : 5/12/21 7:28 PM
FileName : service.py
"""

from bert_serving.client import BertClient
import numpy as np
import pandas as pd
from termcolor import colored
output_file='encoded_quora.npy'
doc_vecs = np.load(output_file)
data_file = 'quora_questions.csv'

df = pd.read_csv(data_file, sep=',', header=0,na_filter=False)
questions = df['question1'].tolist()
topk = 10
print(len(doc_vecs))
print(doc_vecs[:2])

with BertClient(check_length=False) as bc:
    # while True:
        query = input(colored('your question: ', 'green'))
        # query = 'how to get money'
        query_vec = bc.encode([query])[0]
        score = np.sum(query_vec * doc_vecs, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        print('top {} questions to {} :\n'.format(topk, colored(query, 'green')))
        for idx in topk_idx:
            print('>{}\t{}'.format(colored(score[idx],'cyan'), colored(questions[idx],'yellow')))