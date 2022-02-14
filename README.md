# Multiview Embeddings for Soundscape Classification

This repository contains python implementation of our paper "Multiview Embeddings for Soundscape Classification" accepted for publication in the IEEE/ACM Transactions on Audio, Speech and Language Processing Feb 2022.

## Introduction
The work is build on publicly available code for RPCA, L3Net and MvLDAN. We kept all of them here for convenience.

## Getting Started

These instructions will help you to run python programs in sequence.

### Steps to decompose the audio files into the foreground and the background using rpca

1. Download or clone this repository into local system.

#### Development dataset

1. Download DCASE 2017 ASC(task 1) development dataset and extract all the zip files into single folder.
2. Copy all the extracted wav files to folder "<path_to_repo_download>/MultiviewEmbeddingsforSoundscapeClassification/dataset/development/stereo/"
3. Navigate to "<path_to_repo_download>/MultiviewEmbeddingsforSoundscapeClassification/dataset/development/" and run python program "sterio2mono.py".
```
cd <path_to_repo_download>/MultiviewEmbeddingsforSoundscapeClassification/dataset/development/
python sterio2mono.py
```
4. Run the following command to copy the audio files into respective class folders.
```
python copyfiles.py
```
5. Run the following matlab program to decompose all the audio files into foreground and background.
```
cd ../../rpca/development
matlab -nodisplay -r readfilenames
```
6. Run the following commands to copy the foreground and the background into seperate folders.
```
cp -r example/output/*_E.wav ../../dataset/development/rpca_out_foreground/
cp -r example/output/*_A.wav ../../dataset/development/rpca_out_background/
```
7. Run the following commands to remove substring "_A" and "_E" from all the file names.
```
cd ../../dataset/development/rpca_out_foreground/
rename 's/_E//g' *.wav

cd ../rpca_out_background/
rename 's/_A//g' *.wav
```
8. Run the following command to copy the audio files into respective class folders.
```
cd ..
python copyfiles_fg.py
python copyfiles_bg.py
```

#### Evaluation dataset

1. Download DCASE 2017 ASC(task 1) evaluation dataset and extract all the zip files into single folder.
2. Copy all the extracted wav files to folder "<path_to_repo_download>/MultiviewEmbeddingsforSoundscapeClassification/dataset/evaluation/stereo/"
3. Navigate to "<path_to_repo_download>/MultiviewEmbeddingsforSoundscapeClassification/dataset/evaluation/" and run python program "sterio2mono.py".
```
cd <path_to_repo_download>/MultiviewEmbeddingsforSoundscapeClassification/dataset/evaluation/
python sterio2mono.py
```
4. Run the following command to copy the audio files into respective class folders.
```
python copyfiles.py
```
5. Run the following matlab program to decompose all the audio files into foreground and background.
```
cd ../../rpca/evaluation
matlab -nodisplay -r readfilenames
```
6. Run the following commands to copy the foreground and the background into seperate folders.
```
cp -r example/output/*_E.wav ../../dataset/evaluation/rpca_out_foreground/
cp -r example/output/*_A.wav ../../dataset/evaluation/rpca_out_background/
```
7. Run the following commands to remove substring "_A" and "_E" from all the file names.
```
cd ../../dataset/evaluation/rpca_out_foreground/
rename 's/_E//g' *.wav

cd ../rpca_out_background/
rename 's/_A//g' *.wav
```
8. Run the following command to copy the audio files into respective class folders.
```
cd ..
python copyfiles_fg.py
python copyfiles_bg.py
```

### Steps to extract L3-Net features and perform other experiments on L3-Net features

1. Run the following program to generate L3-Net features.
```
python l3net/code/l3_feat_dev.py
python l3net/code/l3_feat_eval.py

```
2. Run the following programs to generate training and evaluation numpy arrays.
```
python l3net/code/data_dev.py
python l3net/code/data_eval.py
```
3. Run the following program to prepare data for background view and foreground view. The amount of foregroud or background to be suppressed can be changed by varying p and q values in the program.
```
python l3net/code/data_prep_mean_nap.py
```

4. Run the following program to train multi-view network to extract embedding from the latent common space for classification.
```
python MvLDAN/main_DCASE_2view.py
```
5. Run the following program to classify using SVM classifier. 
```
python MvLDAN/svm_audio_mean.py
python MvLDAN/svm_avg_mean.py
python MvLDAN/svm_early_fusion_mean.py
```
