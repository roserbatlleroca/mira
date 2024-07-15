#########################################
# MIRA TOOL: DEfNet SCORE ###############
#########################################

# DEfNet score based on track weights of 
# Discogs-Effnet Essentia models 

# v1.0
# last update: june 2024

#########################################
# Import modules ########################
#########################################

import essentia.standard as estd
import torch
import numpy as np
import glob
import csv
import pandas as pd
import os

import argparse
import datetime

#######################################
# DEfNet functions ####################
#######################################

def defnet_eval(folder_A, folder_B, eval_name, log): 
    if log == 'no': 
        LOG_ACTIVE = False 

    elif log is None: 
        LOG_ACTIVE = True
        logdir = 'log'
    
    else: 
        # log folder has been specified 
        LOG_ACTIVE = True
        logdir = log

    # Print computing status
    print("Calculating DEfNet score...")
 
    # Upload Discogs-Effnet model
    print("\nUploading Discogs-Effnet model...\n")

    model_path = "./misc/discogs_track_embeddings-effnet-bs64-1.pb"
    model = estd.TensorflowPredictEffnetDiscogs(graphFilename=model_path, output="PartitionedCall:1")

    # Group A & B 
    audio_file_A = glob.glob(folder_A + '*.wav')
    audio_file_B = glob.glob(folder_B + '*.wav')

    # Compute target probability 
    print("\nGroup B: Getting target probs... \n")
    target_probs = pd.DataFrame(columns = ['songB', 'audio_embed'])
    for s in audio_file_B: 
        audio = estd.MonoLoader(filename=s, sampleRate=16000, resampleQuality=4)()
        audio_embed_B = torch.tensor(model(audio))
        print(s, audio_embed_B)
        target_probs.loc[len(target_probs.index)] = [s, audio_embed_B]

    # Get cosine similarity 
    print("Calculating DEfNet score...")
    matrix_results = []
    c = 0 
    for a in audio_file_A:
        print('{}/{}'.format(c, len(audio_file_A)))
        c += 1
        audio = estd.MonoLoader(filename=a, sampleRate=16000, resampleQuality=4)()
        audio_embed_A = torch.tensor(model(audio))
        for b in audio_file_B: 
            audio_embed_B = target_probs.loc[target_probs['songB']==b, 'audio_embed'].values
            audio_embed_B = audio_embed_B[0].detach().clone()
            cosine_sim = torch.nn.functional.cosine_similarity(audio_embed_A, audio_embed_B, dim=1, eps=1e-8)
            cosine_sum = cosine_sim.sum(dim=0)
            weight = torch.tensor(cosine_sim.size(0))
            dd = cosine_sum/weight
            matrix_results.append([a.split('/')[-1].split('.')[0], b.split('/')[-1].split('.')[0], dd.item()])
            
            
    df = pd.DataFrame(matrix_results, columns=["songA", "songB", "defnet_score"])


    defnet_mean = df['defnet_score'].mean()
    defnet_median = df['defnet_score'].median()
    print("Mean DEfNet score:", defnet_mean)
    print("Median DEfNet score:", defnet_median)

    if LOG_ACTIVE is True: 
        if logdir == 'log': 
            if not os.path.exists(logdir): os.makedirs(logdir)

        # independent file with results
        with open('{}/{}_defnet.csv'.format(logdir, eval_name), 'w') as f: 
            writer = csv.writer(f)
            writer.writerow(['songA', 'songB','defnet_score'])
            for s in range(len(matrix_results)): 
                writer.writerow([matrix_results[s][0], matrix_results[s][1], matrix_results[s][2]])

        # global file with all results 
        if os.path.exists('{}/{}_allresults.csv'.format(logdir, eval_name)): 
            print('\nfile exists: adding DEfNet score results')
            data = pd.read_csv('{}/{}_allresults.csv'.format(logdir, eval_name)) 
            for r in matrix_results: 
                mask = (data['songA'] == r[0]) & (data['songB'] == r[1])
                data.loc[mask, 'defnet_score'] = r[2]
        else: 
            print('\nmissing file: creating new file with DEfNet score results')
            data = df
        
        data.to_csv('{}/{}_allresults.csv'.format(logdir, eval_name), index=False)


