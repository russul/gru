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
    ins_len = len(midi_data.instruments)
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

    # print(np.array(fs).shape)
    # 分别按不同乐器按音符起始时间升序排序
    fs_cut_by_instrument = []
    for i in range(ins_len):
        tmp = [f for f in fs if (f[0] == midi_data.instruments[i].program)]
        fs_cut_by_instrument.append(tmp)

    fs_cut_by_instrument = np.array(fs_cut_by_instrument)


    for j in range(fs_cut_by_instrument.shape[0]):

        # 按起始时间升序排序
        # sorted:第一参数是待排序集合，key:排序规则（使用lamda表示）按第4列.默认升序
        fs_cut_by_instrument[j] = sorted(fs_cut_by_instrument[j],key=lambda x:x[3])

    column_names=['instrument','is_drum','pitch','begin_tick','end_tick','dur_tick','begin_sec','end_sec','dur_sec','vol','bpm','numerator','denominator','key','beats']
    out = (''+ midifile).split('.')


    fs_cut_by_instrument_2D = fs_cut_by_instrument[0]

    x = np.arange(1,ins_len,1)
    for i in x:
        fs_cut_by_instrument_2D = np.row_stack((fs_cut_by_instrument_2D,fs_cut_by_instrument[i]))

    writeExcel(out[0] + '.xlsx',fs_cut_by_instrument_2D,column_names)

    return fs_cut_by_instrument_2D

def skyline_feature(data,path,header):
    data = np.array(data)
    rows = data.shape[0]
    columns = data.shape[1]
    time = np.row_stack((data[:,6],data[:,7]))
    # 去重并排序
    time = np.unique(time)

    idx = []

    for i in range(len(time)-1):
        each_idx = []
        s = time[i]
        e = time[i+1]
        for j in range(rows):
            if(data[j,6]>=s and data[j,6]<e):
                each_idx.append(j)
        idx.append(each_idx)
    rs = []
    row = 0
    for each in idx:
        tmp =[]

        for c in each:
            tmp.append(data[c,:])
        if(len(tmp)==0):
            continue
        tmp = np.array(tmp)
        # 从大到小排序
        tmp = tmp[np.argsort(-tmp[:,3],axis=0)]
        c_rs = tmp[0]
        transfer = c_rs[5]/c_rs[8]
        c_rs[3] = int (time[row]*transfer)
        c_rs[4] = int (time[row+1]*transfer)
        c_rs[5] = c_rs[4]-c_rs[3]
        c_rs[6] = time[row]
        c_rs[7] = time[row+1]
        c_rs[8] = c_rs[7]-c_rs[6]
        c_rs[14]= c_rs[8]* (c_rs[10]/60)
        row+=1
        rs.append(c_rs)
    writeExcel(path,rs,header)
    return rs

    # data_sort_by_time = data[data[:,3].argsort()]

def union_feature(data,path,header):
    data = np.array(data)
    sorted = np.transpose(data[:,2])
    len = sorted.shape[0]
    if(len<=1):
        return data
    repeat = sorted[0]
    cnt = 1
    start = 0
    x = np.arange(1,len,1)
    rs = []
    cut_point = [0]

    for i in x:
        if(sorted[i]==repeat):
            cnt+=1
        else:

            cut_point.append(i)
            cnt = 1
            repeat = sorted[i]

            start = i

    cut_point.append(len)
    rs = union_opreation(data,cut_point)
    writeExcel(path,rs,header)
    return rs

def union_opreation(data,cut_point):
    data = np.array(data)
    cut_point = np.array(cut_point)

    x = np.arange(0,cut_point.shape[0]-1,1)
    rs = []
    for i in x:
        start = cut_point[i]
        end = cut_point[i+1]
        tmp = data[start, :]

        tmp[3] = data[start, 3]
        tmp[4] = data[end - 1, 4]
        tmp[5] = tmp[4] - tmp[3]
        tmp[6] = data[start, 6]
        tmp[7] = data[end - 1, 7]
        tmp[8] = tmp[7] - tmp[6]
        tmp[14] = tmp[8] * (tmp[10] / 60)
        rs.append(tmp)


    return rs


def skyline_feature_v2(data,path,header):
    data = np.array(data)

    data_rows = data.shape[0]
    data_cols = data.shape[1]
    if(data_rows<=1):
        return data
    process = []

    process.append(data[0,:])

    rs = []
    x = np.arange(1,data_rows,1)
    for i in x:
        process.append(data[i,:])

        process,rs = skyline_core(process,rs)


    writeExcel(path,rs,header)
    return rs



def skyline_core(process,rs):
    process = np.array(process)
    rows = process.shape[0]
    cols = process.shape[1]

    flag = np.zeros((1,rows))

    sj = process[rows-1,3] # start_tick
    ej = process[rows-1,4] # end_tick
    pj = process[rows-1,2]
    x1 = np.arange(0, rows - 1, 1)
    for i in x1:
        if(flag[0,i]==0):
            si = process[i,3]
            ei = process[i,4]
            pi = process[i,2]

            if(pi<=pj):

                tmp = np.copy(process[i,:])

                if(ei<=sj):
                    rs.append(tmp)
                    flag[0,i] = 1

                else:

                    tmp[3] = process[i, 3]
                    tmp[4] = process[rows-1, 3]
                    tmp[5] = tmp[4] - tmp[3]
                    tmp[6] = process[i, 6]
                    tmp[7] = process[rows - 1, 6]
                    tmp[8] = tmp[7] - tmp[6]
                    tmp[14] = tmp[8] * (tmp[10] / 60)
                    if(tmp[5]!=0): rs.append(tmp)

                    if(ei<=ej):

                        # 切分当前音符，以便下次循环操作
                        process[rows-1,3] = ei
                        process[rows-1,5] = process[rows-1,4]-process[rows-1,3]
                        process[rows-1,6] = process[i,7]
                        process[rows-1,8] = process[rows-1,7]-process[rows-1,6]
                        process[rows-1,14] = process[rows-1,8]*(process[rows-1,10]/60)

                        # 更新当前音符sj,ej,pj
                        sj = process[rows - 1, 3]  # start_tick
                        ej = process[rows - 1, 4]  # end_tick
                        pj = process[rows - 1, 2]

                    else:
                        # 切分处理音符
                        process[i, 3] = ej
                        process[i, 5] = process[i, 4] - process[i, 3]
                        process[i, 6] = process[rows-1,7]
                        process[i, 8] = process[i, 7] - process[i, 6]
                        process[i, 14] = process[i, 8] * (process[i, 10] / 60)
                        # 标记已处理完音符
                        # flag[0,rows-1] = 1
                        break
            else:
                if(ei<=sj):
                    rs.append(process[i,:])
                    flag[0,i] = 1
                else:
                    if(ei<=ej):
                        # 切分当前音符，以便下次循环操作
                        process[rows-1,3] = ei
                        process[rows-1,5] = process[rows-1,4]-process[rows-1,3]
                        process[rows-1,6] = process[i,7]
                        process[rows-1,8] = process[rows-1,7]-process[rows-1,6]
                        process[rows-1,14] = process[rows-1,8]*(process[rows-1,10]/60)
                    else:
                        flag[0,rows-1] = 1
                        break

    x2 = np.arange(0,rows,1)
    # 更新过程矩阵
    process_change = []
    for k in x2:
        if(flag[0,k]==0 and process[k,5]!=0):
            process_change.append(process[k,:])

    process_change = np.array(process_change)
    process_change=process_change[np.argsort(process_change[:,3],axis=0)]
    process_change = list(process_change)
    return process_change,rs


def select_copy_high(data):
    return
if __name__ == '__main__':
    file = 'Hawaiian.mid'
    base = base_feature(file)
    print(type(base))
    print(base.shape)
    ins_kind = set(base[:,0])
    ins_kind = list(ins_kind)
    features_cut = []
    for i in range(len(ins_kind)):
        tmp = [f for f in base if(f[0]==ins_kind[i])]
        features_cut.append(tmp)

    features_cut = np.array(features_cut)
    print(features_cut.shape[0])
    header = ['instrument','is_drum','pitch','begin_tick','end_tick','dur_tick','begin_sec','end_sec','dur_sec','vol','bpm','numerator','denominator','key','beats']
    for j in range(np.array(features_cut).shape[0]):
        rs = skyline_feature_v2(features_cut[j],'skyline_'+ file.split('.')[0]+'_'+str(j)+'.xlsx',header)
        print(rs)


    # df = pd.read_excel('_3.xlsx')
    # print(df.head(10))
    # print(type(df))
    # data = df.to_numpy()
    #
    # union_feature(data[:,1:16],'n_1.xlsx')







    # features = base_feature('HeyLittl.mid')
    #
    # features = [f for f in features if(f[1]==0)]
    # print(type(features))
    # ins_kind = set([f[0] for f in features])
    # ins_kind = list(ins_kind)
    # features_cut = []
    # for i in range(len(ins_kind)):
    #     tmp = [f for f in features if(f[0]==ins_kind[i])]
    #     features_cut.append(tmp)
    # sk_cut = []
    # for j in range(np.array(features_cut).shape[0]):
    #     sk_cut.append(skyline_feature(features_cut[j],'_'+str(j)+'.xlsx'))
    # # np的axis，三维数组：0：段，1：行(按列操作（max,min,sort）)，2:列(按行操作（max,min,sort）)；在索引时顺序是段行列sk_cut[:,:,2]
    #
    # # py的列表是动态的，索引方式：[][][]...np的数组是固定长度的，索引方式[, , ,]
    # # np的数组可以是任意维度的，但是矩阵必须是2维度
    #
    # avg_pitch=[]
    # # 计算每个乐器的平均音高
    # for sk_cut_i in sk_cut:
    #     col_pitch = [sk[2] for sk in sk_cut_i]
    #     avg_pitch.append(sum(col_pitch)/len(col_pitch))
    # # 选择音高最大的作为主旋律音轨
    # print(avg_pitch)
    # max_pitch_idx = avg_pitch.index(max(avg_pitch))
    # print(max(avg_pitch))
    # print(max_pitch_idx)
    # # 输出乐器编号
    # print(sk_cut[max_pitch_idx][0][0])

    # features = np.array(features)
    # filter_features = []
    # for feature in features:
    #     if(feature[1]==0):
    #         filter_features.append(feature)


    # skyline = skyline_feature(features)

    # duration= features[:,14].T
    # duration=sorted(duration)
    #
    # cols= len(duration)
    # x = np.arange(0,cols,1)
    # print(min(duration))
    # print(max(duration))
    # plt.figure(0)
    # # plt.plot(x,duration_tick,'g',x,duration_sec,'b')
    # plt.subplot(211)
    # plt.plot(x,duration_tick)
    #
    # plt.subplot(212)
    # plt.plot(x,duration_sec)
    # plt.show()
    #
    # plt.figure(0)
    # plt.plot(x,duration,x,0.125+np.zeros((1,cols))[0,:],'b',x,0.125/2+np.zeros((1,cols))[0,:],'g')
    # plt.show()
