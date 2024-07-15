#########################################
# MIRA TOOL: CLAP SCORE #################
#########################################

# CLAP score based on CLAP-LAION-Music 
# Pre-trained model 

# v1.0
# last update: june 2024

#########################################
# Import modules ########################
#########################################

import laion_clap
from laion_clap.training.data import get_audio_features
from laion_clap.training.data import int16_to_float32, float32_to_int16
import glob
import torch
import csv
import os
import librosa
import pandas as pd
import math

import argparse
import datetime

#######################################
# CLAP funcitions #####################
#######################################

def clap_eval(folder_A, folder_B, eval_name, log=None):
    if log == 'no': 
        LOG_ACTIVE = False 

    elif log is None: 
        LOG_ACTIVE = True
        logdir = 'log'
    
    else: 
        # log folder has been specified 
        LOG_ACTIVE = True
        logdir = log
        

    # Upload CLAP pretrained model 
    print("\nUploading CLAP-LAION model...\n")

    # IMPORTANT! Specify the correct path to the pretrained weights 
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    model_path = "./misc/music_audioset_epoch_15_esc_90.14.pt"
    model.load_ckpt(model_path)

    # Otherwise, use default pretrained checkpoints
    # model = laion_clap.CLAP_Module(enable_fusion=False)
    # model.load_ckpt() # download the default pretrained checkpoint.

    # Group A & B 
    audio_file_A = glob.glob(folder_A + '*.wav')
    audio_file_B = glob.glob(folder_B + '*.wav')

    matrix_results = []

    print("\nComputing CLAP score...\n")

    model.eval()

    # Compute embeddings for each pair A-B
    for a in range(len(audio_file_A)): 
        print("{}/{}".format(a, len(audio_file_A)))
        audio_input = []

        # CLAP-LAION interprets up to 10 seconds of audio. For longer audio, it gets
        # a random segment of the input audio. If samples are longer, we divide them 
        # into equal sections smaller than 10 seconds and calculate the embedding for 
        # the total length of the audio instead. 

        # load the waveform of the shape (T,), should resample to 48000
        audio_waveform, _ = librosa.load(audio_file_A[a], sr=48000)   
        if len(audio_waveform) > 480000: # len audio_embed is length (sec) * sr (48000)
            splits = math.ceil(len(audio_waveform)/480000) # How many splits do we need?
            split_length = int(round(len(audio_waveform)/splits, 0))

            offset = 0
            for s in range(splits):                     
                segment = audio_waveform[offset:offset+split_length]
                offset += split_length    
                # quantize
                audio = int16_to_float32(float32_to_int16(segment))
                audio = torch.from_numpy(audio).float()
                temp_dict = {}
                temp_dict = get_audio_features(
                    temp_dict, audio, 480000, 
                    data_truncating='fusion' if model.enable_fusion else 'rand_trunc',
                    data_filling='repeatpad',
                    audio_cfg=model.model_cfg['audio_cfg'],
                    require_grad=audio.requires_grad
                )
                audio_input.append(temp_dict)
        
            audio_embed_A = model.model.get_audio_embedding(audio_input)
        
        else: audio_embed_A = model.get_audio_embedding_from_filelist(x=[audio_file_A[a]], use_tensor=True)

        for b in range(len(audio_file_B)): 
            audio_input = []
            
            # load the waveform of the shape (T,), should resample to 48000
            audio_waveform, _ = librosa.load(audio_file_B[b], sr=48000)   
            if len(audio_waveform) > 480000: # len audio_embed is length (sec) * sr (48000)
                splits = math.ceil(len(audio_waveform)/480000) # How many splits do we need?
                split_length = int(round(len(audio_waveform)/splits, 0))
                offset = 0
                for s in range(splits):                     
                    segment = audio_waveform[offset:offset+split_length]
                    offset += split_length    
                    # quantize
                    audio = int16_to_float32(float32_to_int16(segment))
                    audio = torch.from_numpy(audio).float()
                    temp_dict = {}
                    temp_dict = {}
                    temp_dict = get_audio_features(
                        temp_dict, audio, 480000, 
                        data_truncating='fusion' if model.enable_fusion else 'rand_trunc',
                        data_filling='repeatpad',
                        audio_cfg=model.model_cfg['audio_cfg'],
                        require_grad=audio.requires_grad
                    )
                    audio_input.append(temp_dict)
            
                audio_embed_B = model.model.get_audio_embedding(audio_input)

            else: audio_embed_B = model.get_audio_embedding_from_filelist(x=[audio_file_B[b]], use_tensor=True)

            cosine_sim = torch.nn.functional.cosine_similarity(audio_embed_A, audio_embed_B, dim=1, eps=1e-8)

            cosine_sum = cosine_sim.sum(dim=0)
            weight = torch.tensor(cosine_sim.size(0))
            clap_result = cosine_sum/weight

            matrix_results.append([audio_file_A[a].split('/')[-1].split('.')[0], audio_file_B[b].split('/')[-1].split('.')[0], clap_result.item()])

    df = pd.DataFrame(matrix_results, columns=["songA", "songB", "clap_score"])

    clap_mean = df['clap_score'].mean()
    clap_median = df['clap_score'].median()
    print("Mean CLAP score:", clap_mean)
    print("Median CLAP score:", clap_median)


    if LOG_ACTIVE is True: 
        if logdir == 'log': 
            if not os.path.exists(logdir): os.makedirs(logdir)

        # independent file with results
        with open('{}/{}_clap.csv'.format(logdir, eval_name), 'w') as f: 
            writer = csv.writer(f)
            writer.writerow(['songA', 'songB','clap_score'])
            for s in range(len(matrix_results)): 
                writer.writerow([matrix_results[s][0], matrix_results[s][1], matrix_results[s][2]])
        
        # global file with all results 
        if os.path.exists('{}/{}_allresults.csv'.format(logdir, eval_name)): 
            print('\nfile exists: adding CLAP score results')
            data = pd.read_csv('{}/{}_allresults.csv'.format(logdir, eval_name)) 
            for r in matrix_results: 
                mask = (data['songA'] == r[0]) & (data['songB'] == r[1])
                data.loc[mask, 'clap_score'] = r[2]
        else: 
            print('\nmissing file: creating new file with CLAP score results')
            data = df
        
        data.to_csv('{}/{}_allresults.csv'.format(logdir, eval_name), index=False)

