# coding=utf-8

import numpy as np
import pandas as pd
import pretty_midi
import warnings
import os
import h5py
import multiprocessing
import datetime

# from get_segement import get_phrase_features

def extract_midi_features(path_df, label_dict, max_step):
    """
    This function takes in the path DataFrame, then for each midi file, it extracts certain
    features, maps the genre to a number and concatenates these to a large design matrix to return.

    @input path_df: A dataframe with paths to midi files, as well as their corresponding matched genre.
    @type path_df: pandas.DataFrame

    @return: A matrix of features along with label.
    @rtype: numpy.ndarray of float
    """
    all_features = []
    all_genres = []
    cnt = 0
    total = path_df.shape[0]
    pid = os.getpid()
    for index, row in path_df.iterrows():
        # 自定义get_features !
        #
        # print(row.Path)
        features = get_phrase_features(row.Path)

        genre = label_dict[row.Genre]
        if features is not None:
            f_len = features.shape[1]
            t = np.zeros((max_step, f_len))
            step = features.shape[0]

            if step >= max_step:
                t = features[:max_step]
            else:
                t[:step] = features

            all_features.append(t)
            all_genres.append(genre)
        
        cnt += 1
        if cnt % 10 == 0:
            print("pid: %d, cnt: %d, total:%d" %(pid, cnt, total))

    #return np.array(all_features), np.array(all_genres)
    return all_features, all_genres



def get_phrase_features(path):
    return

def cut_subset_frac(df, genres, factor):
    assert factor > 0 and factor <= 1
    columns = df.keys()
    subset_df = pd.DataFrame(columns=columns)
    
    for g in genres:
        # 提取
        query_df = df.query("Genre == '{}'".format(g))
        # 采样
        sample_df = query_df.sample(frac=factor)
        # 添加
        subset_df = subset_df.append(sample_df, ignore_index=True)
    return subset_df

def cut_subset_N(df, genres, N):
    assert N > 0
    columns = df.keys()
    subset_df = pd.DataFrame(columns=columns)
    
    for g in genres:
        # 提取
        query_df = df.query("Genre == '{}'".format(g))
        # 采样
        if N > query_df.shape[0]:
            sample_df = query_df
        else:
            sample_df = query_df.sample(frac=1).iloc[: N,]
        # 添加
        subset_df = subset_df.append(sample_df, ignore_index=True)
    return subset_df



if __name__ == "__main__":
    # os.system("python get_matched_midi.py")
    # print("matched end")

    max_steps = 50
    n_cpu = 12
    n_sample = 400

    matched_midi_df = pd.read_csv("prepared/matched_midi.csv")

    genre_df = pd.read_csv("prepared/genres.csv")
    label_dict = {row.Genre: idx for idx, row in genre_df.iterrows()}
    genres = genre_df.values.T[0]
    print(genres)

    # 输出目录
    labeled_features_h5 = "prepared/labeled_features.h5"

    # 提取子集
    #subset_df = cut_subset_frac(matched_midi_df, genres, 0.3)

    subset_df = cut_subset_N(matched_midi_df, genres, n_sample)
    print(subset_df.shape)
    
    # 多进程提取特征
    df_lst = []
    step = subset_df.shape[0] // n_cpu
    if subset_df.shape[0] % n_cpu != 0:
        step += 1

    for i in range(n_cpu):
        df_lst.append(subset_df.iloc[i*step: (i+1)*step,])

    # res = extract_midi_features(subset_df, label_dict, max_steps)

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    res_feature = []
    for i in range(n_cpu):
        res_feature.append(pool.apply_async(extract_midi_features, args=(df_lst[i], label_dict, max_steps)))

    pool.close()

    t_start = datetime.datetime.now()
    pool.join()
    t_end = datetime.datetime.now()
    print("total time: %d" % (t_end - t_start).seconds)

    # 整合特征
    all_features = []
    all_genres = []
    for i in range(n_cpu):
        ft_lst = res_feature[i].get()[0]
        lb_lst = res_feature[i].get()[1]
        for ft in ft_lst:
            all_features.append(ft)
        for lb in lb_lst:
            all_genres.append(lb)

    with h5py.File(labeled_features_h5, "w") as f:
        f.create_dataset("features", data=all_features)
        f.create_dataset("labels", data=all_genres)

    '''
    with h5py.File(matched_midis_h5, "r") as fr:
       for k in fr.keys():
           print(fr[k].name)
           print(fr[k].shape)

       label_arr = np.array(fr["labels"])
       label_list = label_arr.tolist()
       label_dict = {lbl: label_list.index(lbl) for lbl in label_list}
       label_cnt = {lbl: 0 for lbl in label_list}
       
       
       matched_midi_df = pd.DataFrame(fr["matched_midis"].value, columns=["Path", "Genre"])
       matched_midi = fr["matched_midis"]

       for i in matched_midi:
           label_cnt[i[1]] += 1

       print(label_cnt)
    '''
            
   
