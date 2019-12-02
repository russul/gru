import pretty_midi as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 将数据写入Excel，注意有大小的限制
def writeExcel(path,data,column_names):

    exceldata = pd.DataFrame(data)
    writer = pd.ExcelWriter(path)
    exceldata.to_excel(writer, 'page_1', float_format='%.4f',header=column_names)  # ‘page_1’是写入excel的sheet名

    # exceldata.to_excel(writer, 'page_1', header=columns)  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()
def base_feature(midifile):

    midi_data = pm.PrettyMIDI(midifile)

    print(len(midi_data.instruments))

    print(midi_data.time_to_tick(0))
    print(midi_data.time_to_tick(9.5))
    print(midi_data.tick_to_time(10000))
    # 调号变化事件
    print(midi_data.key_signature_changes)
    # 拍号变化事件
    print(midi_data.time_signature_changes)
    # 速度变化事件
    print(midi_data.get_tempo_changes())
    # 每个采样点速度估计
    print(midi_data.estimate_tempi())
    # 整个midi的速度估计
    print(midi_data.estimate_tempo())
    # print(midi_data.instruments)


    print(midi_data.instruments)
    fs1 = []
    #
    # tempo = midi_data.get_tempo_changes()
    cnt = []
    for ins in midi_data.instruments:
        i=0
        for note in ins.notes:
            tmp = []
            tmp.append(ins.program)
            tmp.append(ins.is_drum)
            tmp.append(note.pitch)
            tmp.append(midi_data.time_to_tick(note.start))
            tmp.append(midi_data.time_to_tick(note.end))
            tmp.append(midi_data.time_to_tick(note.end-note.start))
            tmp.append(note.start,)
            tmp.append(note.end,)
            tmp.append(note.end-note.start)
            tmp.append(note.velocity)

            fs1.append(tmp)
            i+=1;
        cnt.append(i)
    print(cnt)

    data_rows = sum(cnt)
    fs2 = []
    change_times,tempo = midi_data.get_tempo_changes()
    time_changes = midi_data.time_signature_changes
    key_changes = midi_data.key_signature_changes

    print(tempo)
    time_1 = 4;
    time_2 = 4;
    key = 0;
    tempo_v = 120;
    if len(tempo)>0:
        tempo_v = tempo[0]
    if len(time_changes)>0:
        time_1 = time_changes[0].numerator
        time_2 = time_changes[0].denominator
    if len(key_changes)>0:
        key = key_changes[0].key_number
    for i in range(data_rows):
        tmp = []
        tmp.append(tempo_v)
        tmp.append(time_1)
        tmp.append(time_2)
        tmp.append(key)

        fs2.append(tmp)


    print(fs2)

    fs1 = np.array(fs1)
    print(fs1.shape)
    # fs2 = np.transpose(fs2)
    # print(fs2.shape)
    print(fs2)
    fs = np.column_stack((fs1,fs2))

    print(fs)


    # 添加beats
    beats=fs[:,8]*(fs[:,10]/60)

    fs = np.column_stack((fs,beats))

    print(np.array(fs).shape)
    column_names=['instrument','is_drum','pitch','begin_tick','end_tick','dur_tick','begin_sec','end_sec','dur_sec','vol','bpm','numerator','denominator','key','beats']
    out = (''+ midifile).split('.')

    writeExcel(out[0] + '.xlsx',fs,column_names)

    return fs

if __name__ == '__main__':
    base_feature('HeyLittl.mid')