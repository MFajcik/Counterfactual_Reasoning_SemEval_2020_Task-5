import csv
import numpy as np
import pandas as pd
import regex as re
import logging
import argparse
import json
import time
import os
import matplotlib as mpl
import pickle
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
tok=BertTokenizer.from_pretrained("bert-base-uncased")
input_file="Subtask-1-master/train.csv"
#input_file="Subtask-1-master/train.csv"
A_dict={}
max_seq_len=100
too_long=0
with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    lines = []
    for line in reader:
        A_dict[line[0]]={'label':line[1],'sent':line[2]}
lens=[]
labels=[]
for v in A_dict.values():
    lens.append(len(tok.tokenize(v['sent'])))
    if len(v['sent'].split(" "))>200:
        print(v['sent'].split(" "))
    if len(v['sent'].split(" "))>max_seq_len:
        too_long+=1

    labels.append(v['label'])
print ("labeled 0: %s"%labels.count('0'))
print ("labeled 1: %s"%labels.count('1'))

#print(lens1)
#print(lens2)
#print(lens_concat)
print ("longest: %s" %sorted(lens)[-30:])

plt.hist(np.asarray(lens),range=(0,105),bins=53, stacked=True, color='#0504aa',
                            alpha=0.7, rwidth=0.9)
plt.grid(axis='y', alpha=0.25)
plt.xlabel("Count")
plt.xlim(0,105)
plt.ylabel('Tokens')
plt.savefig("asdfg1.png", dpi=320)
print("%s sentences are too long to fit with max seq len of %s"%(too_long,max_seq_len))