#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author   : MuyunLi
Date     : 5/12/21 6:22 PM
FileName : embedding.py
"""

import pandas as pd
import numpy as np
from bert_serving.client import BertClient

output_file = 'encoded_quora.npy'
data_file = 'quora_questions.csv'

df = pd.read_csv(data_file, sep=',', header=0,na_filter=False)


questions=df['question1'].tolist()[:1000]

print(len(questions))
with BertClient(check_length=False) as bc:
    print('questions encoded..')
    doc_vecs = bc.encode(questions)
    print(type(doc_vecs))

    np.save(output_file, doc_vecs)
    print('saved..')



