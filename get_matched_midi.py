# coding=utf-8

import numpy as np
import pandas as pd
import h5py
import os


if __name__ == "__main__":
    midi_dir = "midi"
    midi_lst = []
    genre_lst = []
    genres = None
    for dirpath, dirnames, filenames in os.walk(midi_dir):
        if genres == None:
            genres = dirnames.copy()
            print(dirnames)
        for f in filenames:
            _, genre = os.path.split(dirpath)
            full_path = os.path.join(dirpath, f)
            midi_lst.append(full_path)
            genre_lst.append(genre)


    matched_midi_df = pd.DataFrame({"Path": midi_lst, "Genre": genre_lst})
    matched_midi_df.to_csv("prepared/matched_midi.csv", index=False)

    genre_df = pd.DataFrame({"Genre":genres})
    print(genre_df)
    genre_df.to_csv("prepared/genres.csv", index=False)

    '''
    with h5py.File("prepared/matched_midi.h5", "w") as f:
        dt = h5py.special_dtype(vlen=str)
        ds_matched_midi = f.create_dataset("matched_midi", matched_midi_df.shape, dtype=dt)
        ds_matched_midi[:] = matched_midi_df.values
    '''
