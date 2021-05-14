# Spam classification using VAD

# Project Title:
Spam user detection in crowdsourced speech data via Voice Activity Detection using Deep NN.

# Project Description:
Input : Input to the model would consist of  an audio file in .wav format of around 5-10 sec.
This is generally the average length of the recorded sound files.
Output:  The model outputs a label 0/1 depending upon whether speech is present in the wav file or not

# Evaluation Metrics: 
The evaluation metric in our case will be classification accuracy of the model calculated on the basis of true positive, false positives, true negative and false negative


# Methodology

Voice activity detection (VAD) refers to the task of determining whether a signal contains speech or not. It is thus a binary decision. For an input signal x, our objective is to determine whether it is speech or not. We express the VAD algorithm as a function y=VAD(x), where the desired target output is, y = 0 if only noise else y= 1

# DataSets
## Libri-Speech Dataset: 
LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned. We have used a clean speech dataset from it for our task.
## QUT Noise Dataset: 
This is a huge dataset which contains different types of noise from CAFE, CAR, STREET etc. This noise dataset is being used to create a new dataset for the purpose.
## CLAP Dataset: 
CLAP is an initiative at the Department of Computer Science at IIT Bombay to collect labeled speech to build speech-driven technologies for Indian languages. This is the real time dataset which has background noise in it along with speech in multiple indian languages. A special characteristic of this data-set is that we have the real time feedback  from users on the audio-quality with parameters like “Audio is fine” (AIF) ,”Background is Noisy” (BIN) and so on. We plan to use this effectively to train our models.
## Blind test set: 
The blind test set consists of data from a manually detected spam CLAP user (contains around 200 wav files only containing noise.)  Speech recordings from a few users with low quality voice and high background noise has also been filtered out to serve as an evaluation metric on how our model performs on borderline cases.
# Challenges:
Our main challenge is classification of audio files which are purely noise from the ones which have speech content as well. In our clap data we have various files which are noisy but have speech content, sometimes difficult to distinguish from pure noise and the existing models get confused and misclassify noise as speech and vice versa. Some of the users have a tendency to spam the data (record blank /noisy recordings without any speech content) .We wish to make a tool which identifies such users on an early basis so that we can maintain the quality of our data without manual intervention. This can also be used for other crowdsourced tools which collect such crowdsourced user data-sets. 

# Meta-data description and notebook links for collab editing

More information about meta-data and direct links for editing here :
https://docs.google.com/document/d/1kEIEj0f812N1YcOX1PmRGi172H6myZz0wuPK8eycmFQ/edit

Data folder link (contains all kinds of wav files and model checkpoints, both processed and non-processed) :
https://drive.google.com/drive/folders/1dovc8A9Q0Yu-3j97BcX_N4CGIbnUfGMF?usp=sharing


