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
    tags_set = map(lambda k: k.replace("'", "$").replace('"', "@"), tags_set)
    dat = df.values.tolist()
    dat.sort(key=lambda k: k[-1])
    # random.shuffle(dat)
    nums = len(dat)
    nums_train = int(nums *  0.8)
    index = 0
    headers = df.columns.values.tolist()[:-1]
    f_train = open("data/movielens_train", "wb")
    f_train.write("|".join(["target"] + headers) + "\r\n")
    f_test = open("data/movielens_test", "wb")
    f_test.write("|".join(["target"] + headers) + "\r\n")
    for line in dat:
        line = line[:-1]
        line[-1] = line[-1].replace("'", "$").replace('"', "@")
        real_tag = line[2]
        positive_sample = "|".join(['1'] + line) + "\r\n"
        tag = random.choice(tags_set)
        while tag == real_tag:
            tag = random.choice(tags_set)
        line[2] = tag
        negtive_sample_1 = "|".join(['0'] + line)+ "\r\n"
        tag = random.choice(tags_set)
        while tag == real_tag or tag == line[2]:
            tag = random.choice(tags_set)
        line[2] = tag
        print line
        negtive_sample_2 = "|s".join(['0'] + line)+ "\r\n"
        if index < nums_train:
            f_train.writelines([positive_sample, negtive_sample_1, negtive_sample_2])
        else:
            f_test.writelines([positive_sample, negtive_sample_1, negtive_sample_2])
        index += 1
    f_train.close()
    f_test.close()






if __name__ == "__main__":
    main()