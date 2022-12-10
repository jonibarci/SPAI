#!/Users/keremokyay/miniforge3/envs/spai/bin/python
import numpy as np
import librosa 
import sklearn
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import music21
from scipy import signal

def add_to_note_info(cps,W,H,S,segment,sr, note_info,s,onset_times):
    for n in range(cps):
        spectral = W[:,n]
        temporal = H[n]
        #_onsets += librosa.util.peak_pick(temporal,pre_max = 6, 
                                        #post_max=6,pre_avg=50, post_avg=50,
                                        #delta=temporal.max()/2,wait=0)

        #n = librosa.hz_to_midi(np.argmax(spectral) * (sr/2)/S.shape[0])
        note = np.argmax(spectral) * (sr/2)/S.shape[0]
        note_info += [[(segment-2)+round(onset_times[n],2),round(note)]]
        #for o in onsets_:
            #note_info += [[(segment-2)+round(o/S.shape[1],3),round(n)]]
    #if len(_onsets) == len(onset_times):
        #for o,n in zip(_onsets,_notes):
            #note_info+=[[o,n]]
    #else:
        #pass



    note_info = sorted(note_info)
    last_onset = 0
    for i in range(len(note_info)-1):
        if(len(note_info[i]) == 3):
            continue
        duration = note_info[i+1][0] - note_info[i][0] 
        note_info[i] += [round(duration,2)]
        last_onset = note_info[i][0]

    if len(note_info) != 0:
        note_info[-1] += [round(segment-last_onset,2)]

        #for i in note_info:
            #n = music21.note.Note(i[1]) 
            #s.append(n)

    return note_info,s
        



def window(duration,x,sr):
    note_info = []
    s = music21.stream.Stream()
    
    for i in range(2,int(duration),2):
        curr_x = x[(i-2)*sr:sr*i] 
        S = librosa.stft(curr_x,n_fft=int(sr/8))
        #D = np.abs(librosa.iirt(curr_x))
        ##C = np.abs(librosa.cqt(y=curr_x, sr=sr))
        onset_frames = librosa.onset.onset_detect(curr_x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
        onset_times = librosa.frames_to_time(onset_frames)
        cps = len(onset_times)
        if cps == 0:
            continue
        X, X_phase = librosa.magphase(S)
        W, H = librosa.decompose.decompose(X,n_components=cps, sort=True)
        note_info,s = add_to_note_info(cps,W,H,S,i,sr,note_info,s,onset_times)
    return note_info,s




if __name__=='__main__':
    x,sr = librosa.load("/Users/keremokyay/masters/SPAI/Project-sessions/data/drum_mic2.wav")
    sos = signal.butter(10, 15, 'hp', fs=sr/2, output='sos')
    filtered = signal.sosfilt(sos, x) 
    duration = len(x) / sr

    tempo, beats = librosa.beat.beat_track(y=x,sr=sr)
    tempo=int(2*round(tempo/2))
    mm = music21.tempo.MetronomeMark(referent='quarter', number=tempo)
    note_info,s = window(duration,x,sr)
    s.append(mm)
    onset_frames = librosa.onset.onset_detect(x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    onset_times = librosa.frames_to_time(onset_frames)
    #print(note_info)
    for i in note_info:
        print(i)
    print(len(note_info))
    print(len(onset_times))
    
    #s.write('midi', fp='midi_drum.mid')




