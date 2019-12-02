import pretty_midi as pm
import pandas as pd
import numpy as np



# 将数据写入Excel，注意有大小的限制
def writeExcel(path,data):

    exceldata = pd.DataFrame(data)
    writer = pd.ExcelWriter(path)
    exceldata.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()



def test(rs):
    rs.append([1])



rs = []

test(rs)

print(rs)


# midi_data = pm.PrettyMIDI('baga01.mid')
# for inst in midi_data.instruments:
#
#     print(inst.get_onsets())
#
#
# onsets = midi_data.instruments[0].get_onsets()
# end = midi_data.instruments[0].get_end_time()
# print(len(onsets))
# print(end)


# midi_data = pm.PrettyMIDI("HeyLittl.mid")
# # 返回音符种类的直方图（一共12种音符），可以选择归一化
# print(midi_data.get_pitch_class_histogram())
# # 音符转移矩阵
# print(midi_data.get_pitch_class_transition_matrix())
# for inst in midi_data.instruments:
#     print(inst.get_pitch_class_histogram())
#     print(inst.get_pitch_class_transition_matrix())

# print(len(midi_data.instruments))
#
# print(midi_data.time_to_tick(0))
# print(midi_data.time_to_tick(9.5))
# print(midi_data.tick_to_time(10000))
# # 调号变化事件
# print(midi_data.key_signature_changes)
# # 拍号变化事件
# print(midi_data.time_signature_changes)
# # 速度变化事件
# print(midi_data.get_tempo_changes())
# # 每个采样点速度估计
# print(midi_data.estimate_tempi())
# # 整个midi的速度估计
# print(midi_data.estimate_tempo())
# print(midi_data.instruments)


# fs1 = []
#
# tempo = midi_data.get_tempo_changes()
# cnt = []
# for ins in midi_data.instruments:
#     i=0
#     for note in ins.notes:
#         tmp = []
#         tmp.append(ins.program)
#         tmp.append(note.pitch)
#         tmp.append(midi_data.time_to_tick(note.start))
#         tmp.append(midi_data.time_to_tick(note.end))
#         tmp.append(midi_data.time_to_tick(note.end-note.start))
#         tmp.append(round(note.start,4))
#         tmp.append(round(note.end,4))
#         tmp.append(round(note.end-note.start,4))
#         tmp.append(note.velocity)
#
#         fs1.append(tmp)
#         i+=1;
#     cnt.append(i)
# print(cnt)
#
# data_rows = sum(cnt)
# fs2 = []
# change_times,tempo = midi_data.get_tempo_changes()
# print(tempo)
# for i in range(data_rows):
#     fs2.append(round(tempo[0]))
#
#
# print(fs2)
#
# fs1 = np.array(fs1)
# print(fs1.shape)
# # fs2 = np.transpose(fs2)
# # print(fs2.shape)
# print(fs2)
# fs = np.column_stack((fs1,fs2))
#
# print(fs)
# # for i in range(len(fs1)):
# #     print(fs1[i])
#
# print(midi_data.get_beats())
# print(midi_data.get_downbeats())

# for ins in midi_data.instruments:
    # print(ins.name)
    # 返回该乐器所有play（发声）的音符，起始的时间（可能有重复的值，因为像钢琴这样的乐器可以同时发出多个音符）
    # print(ins.get_onsets())
    # 返回音符
    # print(ins.notes)
    # for note in ins.notes:
    #     # 音符四个属性：起始时间，结束时间，音符编号（0-127），速度(根据事件不同，这里表示的是不同的意思，对于音符开始时间来说是力度（音量）0-127)
    #     print(note)


    # print(ins.pitch_bends)
    # print(ins.control_changes)
    # 返回钢琴卷帘图
    # print(ins.get_piano_roll())

# for instrument in midi_data.instruments:
#     for note in instrument.notes:
#
#         print(note)
#     print('---------------------------------------------')




