# —*- coding:utf-8 -*-
__author__ = "zengpan"

import pandas as pd
import numpy as np
import random

RAW_DATA_PATH = "raw_data/tags.csv"



def main():
    df = pd.read_csv(RAW_DATA_PATH, dtype=str)
    df = df.dropna(axis=0, how='any')
    tags_set = list(set(df['tag'].values))
    dat = df.values.tolist()
    dat.sort(key=lambda k: k[-1])
    with open("data/movielens.csv", "wb") as f:
        for line in dat:
            real_tag = line[2]
            positive_sample = ",".join(line + ['1']) + "\r\n"
            tag = random.choice(tags_set)
            while tag == real_tag:
                tag = random.choice(tags_set)
            line[2] = tag
            negtive_sample_1 = ",".join(line + ['0'])+ "\r\n"
            tag = random.choice(tags_set)
            while tag == real_tag or tag == line[2]:
                tag = random.choice(tags_set)
            line[2] = tag
            print line
            negtive_sample_2 = ",".join(line + ['0'])+ "\r\n"
            f.writelines([positive_sample, negtive_sample_1, negtive_sample_2])






if __name__ == "__main__":
    main()