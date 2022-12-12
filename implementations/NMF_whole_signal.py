#!/Users/keremokyay/miniforge3/envs/spai/bin/python
import numpy as np
import re
import math
import pandas as pd
import librosa 
import music21


def add_to_note_info_whole(cps,W,H,S,sr, note_info,s,onset_times,duration):
    # for every component
    temp_note_info = []
    for n in range(cps):
        # get the coresponding spectral profile
        spectral = W[:,n]
        # get the corresponding temporal profile
        temporal = H[n]

        # get the highest activation in the spectral profile
        note = np.argmax(spectral) * (sr/2)/S.shape[0]
        # get the highest activation in temporal profile
        onset = np.argmax(temporal)/S.shape[1]
        # add the info to the temporary array
        temp_note_info += [[onset,n,note]]

    # sort the array based on increasing onset time
    temp_note_info= sorted(temp_note_info)
    # sort the onsets from Wannes' code
    onset_times = sorted(onset_times)
    # use the info in temp array to index Wannes' onsets
    for info in temp_note_info:
        # error correct the onsets, use the estimated onset from H to index the onsets from
        # Wannes' code
        note_info += [[round(onset_times[info[1]],2),round(info[2])]]

    # calculate durations
    note_info = sorted(note_info)
    last_onset = 0

    # until last element, loop through note info
    for i in range(len(note_info)-1):
        # next element - current element is the duration of the current note
        duration_ = note_info[i+1][0] - note_info[i][0] 
        # add it to the note_info
        note_info[i] += [round(duration_,2)]
        # update the last onset
        last_onset = note_info[i][0]

    # check if the note  info aray has elements
    if len(note_info) != 0:
        # using the last onset handle the last element in the note info array
        note_info[-1] += [round(duration-last_onset,2)]

        # append the notes to the music stream
        # TODO: adding the duration and other attributes for the midi file
        for i in note_info:
            n = music21.note.Note(i[1]) 
            s.append(n)

    return note_info,s

        
def apply_on_whole(x,sr):
    note_info = []
    s = music21.stream.Stream()
    
    curr_x = x 
    # apply STFT
    S = librosa.stft(curr_x)

    onset_envelope = librosa.onset.onset_strength(y=curr_x, sr=sr, hop_length=512)
    onsets = librosa.onset.onset_detect(y=curr_x, sr=sr, onset_envelope=onset_envelope, hop_length=512)
    #onset_frames = librosa.onset.onset_detect(curr_x, sr=sr, wait=1, pre_avg=1,hop_length=256, post_avg=1, pre_max=1, post_max=1)
    onset_times = librosa.frames_to_time(onsets)

    print(onset_times)

    # estimate reduced rank R for NMF
    cps = len(onset_times)
    # if there are no onsets continue to the next iteration
    print("R: ",cps)

    # decompose spectrogram S to magnitude and phase
    X, X_phase = librosa.magphase(S)
    # use the Magnitude spectrum of S to do NMF, fit=True, components are estimated from X
    W, H = librosa.decompose.decompose(X,n_components=cps)

    duration  = len(x)/sr
    print(duration)
    note_info,s = add_to_note_info_whole(cps,W,H,S,sr, note_info,s,onset_times,duration)

    return note_info,s




if __name__=='__main__':
    path = "/Users/keremokyay/masters/SPAI/Project-sessions/data/five.wav"
    test_path = "/Users/keremokyay/Documents/labeled_data_SPAI/part1.txt"
    f = open(test_path)
    labels = []
    for line in f:
        line = line.strip('\n')
        row = re.split(';', line)
        row = row[:-1]
        for i in range(3):
            if i == 2:
                row[i] = int(row[i])
            elif len(row[i]) >=7:
                row[i] = row[i][:-4]
                row[i] = round(float(row[i]),2)
        labels += [row]
    # load the file
    x,sr = librosa.load(path)
    
    # calculate duration to loop through
    duration = len(x) / sr

    # calculate tempo 
    tempo, beats = librosa.beat.beat_track(y=x,sr=sr)
    tempo=int(2*round(tempo/2))
    # create the music21 object for tempo
    mm = music21.tempo.MetronomeMark(referent='quarter', number=tempo)

    # start the transcription
    note_info,s = apply_on_whole(x,sr)

    s.append(mm)
    print(note_info)


    # find the onsets that are in +- 0.1 
    correct_onsets = 0
    correct_onsets_within_02 = 0

    correct_notes = 0
    correct_notes_within_20 = 0

    correct_duration = 0
    correct_duration_within_01 = 0

    for label in labels:
        onset = label[0]
        duration = label[1] - label[0]
        note = label[2]

        for prediction in note_info:
            onset_pred = prediction[0]
            duration_pred = prediction[1] - prediction[0]
            note_pred = prediction[2]

            if round(onset_pred,1) == round(onset,1):
                correct_onsets += 1
            elif np.abs(onset - onset_pred) <= 0.2: 
                correct_onsets_within_02 +=1
            if note == note_pred:
                correct_notes += 1
            elif np.abs(note-note_pred) <=20:
                correct_notes_within_20 += 1
            if duration == duration_pred:
                correct_duration += 1
            elif np.abs(note-note_pred) <=0.1:
                correct_duration_within_01 += 1

    print("total number of predicted onsets:      ", len(note_info), "total number of ground truth onsets: ", len(labels))
    print("number of correct onsets predicted:    ", correct_onsets, " number of onsets within 0.2 seconds of the ground truth: ", correct_onsets_within_02)
    print("number of correct predicted notes:     ", correct_notes, " number of notes within 20 Hz of the ground truth: ", correct_notes_within_20 )
    print("number of correct predicted duraitons: ", correct_duration, " number of durations within 0.1 seconds of the ground truth: ", correct_duration_within_01 )
    
    #s.show()
    #s.write('midi', fp='midi_drum.mid')




