#!/Users/keremokyay/miniforge3/envs/spai/bin/python
import numpy as np
import librosa 
import sklearn
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
import music21
from scipy import signal

## Signal parametes
fs = 22050      # sample frequency

## STFT parameters
N = 1024        # frame size
q = 0.5         # overlap factor

## NINOS2 parameters
lambda_ = 1     # compression ratio
gamma_ = 1      # fraction of frequency bins used

## peak picking parameters 
alpha_ = 1      # before maximum
beta_ = 1       # after maximum
a = 3           # before average
b = 2           # after average
delta_ = 1     # offset from neighboorhood average

def ninos_func(X, lambda_, gamma_):

    ## Calculate number of frequency bins
    J = int(gamma_*(X.shape[0]-1)) 

    ## Sort lower J log values of X
    Y = np.zeros((J, X.shape[1]))
    Y_temp = np.sort(np.log10(lambda_*np.abs(X)+1),axis=0)
    Y = Y_temp[0:J,:]

    ## Inverse sparsity
    ninos = np.zeros(Y.shape[1])
    for i in range(Y.shape[1]):
        #ninos[i] = (np.linalg.norm(Y[:,i])**2)/(np.power(J,1/4)*sum(abs(Y[:,i])**4)**(1./4))  
        ninos[i] = np.linalg.norm(Y[:,i])*((np.linalg.norm(Y[:,i])/sum(abs(Y[:,i])**4)**(1./4))-1)/(np.power(J,1/4)-1)

    return ninos

def peak_pick(ninos, alpha_, beta_, a, b, delta_, N, q):
    
    ## Calculate combination width
    Theta_ = int(np.ceil(N/(np.rint((1-q)*N))))
    
    return librosa.util.peak_pick(ninos, alpha_,beta_,a,b,delta_,Theta_)


def add_to_note_info(cps,W,H,S,segment,sr, note_info,s,onset_times):
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
        note_info += [[(segment-2)+round(onset_times[info[1]],2),round(info[2])]]

    # calculate durations
    note_info = sorted(note_info)
    last_onset = 0

    # until last element, loop through note info
    for i in range(len(note_info)-1):
        # if i has already 3 elements, it means duration was calculated in the previous segment
        # continue to the next iteration
        if(len(note_info[i]) == 3):
            continue
        # next element - current element is the duration of the current note
        duration = note_info[i+1][0] - note_info[i][0] 
        # add it to the note_info
        note_info[i] += [round(duration,2)]
        # update the last onset
        last_onset = note_info[i][0]

    # check if the note  info aray has elements
    if len(note_info) != 0:
        # using the last onset handle the last element in the note info array
        note_info[-1] += [round(segment-last_onset,2)]

        # append the notes to the music stream
        # TODO: adding the duration and other attributes for the midi file
        for i in note_info:
            n = music21.note.Note(i[1]) 
            s.append(n)

    return note_info,s
        



def window(duration,x,sr):
    # variables to return when transcription is done
    note_info = []
    s = music21.stream.Stream()
    
    # take 2 second segments from the recording and apply NMF on them
    for i in range(2,int(duration),2):
        # take the next 2 second segment
        curr_x = x[(i-2)*sr:sr*i] 
        # apply STFT
        S = librosa.stft(curr_x,n_fft=int(sr/8))


        # Code from Wannes for onset detection
        t = librosa.times_like(S,hop_length=N)
        odf = ninos_func(S, lambda_, gamma_)
        onsets = peak_pick(odf, alpha_, beta_, a, b, delta_, N, q)
        onset_times = librosa.frames_to_time(onsets)

        # estimate reduced rank R for NMF
        cps = len(onset_times)
        # if there are no onsets continue to the next iteration
        if cps == 0:
            continue
        # decompose spectrogram S to magnitude and phase
        X, X_phase = librosa.magphase(S)
        # use the Magnitude spectrum of S to do NMF, fit=True, components are estimated from X
        W, H = librosa.decompose.decompose(X,n_components=cps, sort=True, fit=True)
        # using W and onset times, populate the note_info
        note_info,s = add_to_note_info(cps,W,H,S,i,sr,note_info,s,onset_times)
    return note_info,s




if __name__=='__main__':
    # load the file
    x,sr = librosa.load("/Users/keremokyay/masters/SPAI/Project-sessions/data/drum_mic.wav")
    
    # calculate duration to loop through
    duration = len(x) / sr

    # calculate tempo 
    tempo, beats = librosa.beat.beat_track(y=x,sr=sr)
    tempo=int(2*round(tempo/2))
    # create the music21 object for tempo
    mm = music21.tempo.MetronomeMark(referent='quarter', number=tempo)

    # start the transcription
    note_info,s = window(duration,x,sr)

    s.append(mm)
    onset_frames = librosa.onset.onset_detect(x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
    onset_times = librosa.frames_to_time(onset_frames)
    #print(note_info)
    for i in note_info:
        print(i)
    print(len(note_info))
    print(len(onset_times))
    
    #s.show()
    #s.write('midi', fp='midi_drum.mid')




