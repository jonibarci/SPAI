[Full Paper](ARINA%20Paper.pdf)


# Introduction

In this paper, we will be addressing the problem of music information
retrieval (MIR) in the context of music therapy. Music therapy is a form
of treatment that utilizes music and musical elements, as well as
instrumental improvisation, to facilitate communication, relationships,
and expression. It is typically led by a trained therapist and conducted
in a group setting with patients. In our case, the group of patients
consists of children with ASD.

We aim to use signal processing tools to automate the analysis of these
sessions, with the ultimate goal of reducing the time it takes for
therapists to classify a child as having ASD or not, and to help them
increase the effectiveness of the treatment.

We begin by providing an overview of related works and state-of-the-art
methods in the field. We then proceed to discuss the three main stages
of our proposed process: blind source separation, onset detection and
automatic music transcription. We explore approaches including principal
component analysis (PCA), independent component analysis (ICA),
NINOS$^2$ algorithm and Non-Negative Matrix Factorization (NMF).

# Conclusion 

This paper represents an approach which utilizes a combination of blind
source separation, note onset and non-negative matrix factorization to
identify and classify musical recordings. The pipeline is structured
such that input recording goes into blind source separation and then
into note onset detection and then into non-negative matrix
factorization. No meaningful performance analysis can be made about our
blind source separation method due to the obstacles such as the lack of
labeled data, non-linearity in the mixing and implementation mistakes in
our Python script. Note onset detection using NINOS method tested on a
small dataset generated from the MAPS dataset which contains both
polyphonic and monophonic music pieces resulted around an F-score of 0.9
for monophonic and 0.8 for polyphonic. The performance analysis of
non-negative matrix factorization is made with two different recordings:
a recording of a piano recording containing 5 isolated notes starting
from C4 until G4 and a recording of 130 seconds long music therapy
session. The result obtained from our algorithm is compared with labeled
data. The number of onsets exactly corresponded to the number of notes
played in the piano recording, however, this is not the case for music
therapy session recording. For music therapy session recording, our
algorithm has 51.16%, 11.26% and 4.04% for onset, note and offset
accuracy respectively. To sum up, our approach performs better on
recordings of isolated instruments than polyphonic music recordings.

Future work can focus on instrument recognition to estimate which
instrument corresponds to which source in the recording. It is also
interesting to approach automatic music transcription in a different way
by making it genre specific instead of generalizing all genres, by this
way more powerful models can be used in estimation of prediction of
results. Another possible approach is by using machine learning/neural
networks, however, machine learning approach is usually treated more or
less as a black box model.
