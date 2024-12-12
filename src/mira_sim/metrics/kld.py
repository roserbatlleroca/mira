#########################################
# MIRA TOOL: KL DIV #####################
#########################################

# KL Divergence: Kullback-Leibler Divergence

# v1.1
# last update: dec 2024

#########################################
# Import modules ########################
#########################################

import csv
import os
import numpy as np
import glob
import torch
import librosa
import pandas as pd
from .kldiv import kld_metric

import argparse
import datetime

#######################################
# KL Divergence functions #############
#######################################

def kld_eval(folder_A, folder_B, eval_name, log, prelen):
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
    print("Calculating KL divergence...")

    # Getting songs in the directory 
    song_list_A = glob.glob(folder_A + '*.wav')
    song_list_B = glob.glob(folder_B + '*.wav')

    kldiv_ab = []
    kldiv_ba = []
    all_kldiv = []

    # Define model 
    model = kld_metric.PasstKLDivergenceMetric(pretrained_length=prelen)

    # Get A songs waveform characteristics
    ref_data = []
    print("\nLoading baseline group (A)...")
    for s in song_list_A: 
        # Load audio 
        y, sr = librosa.load(s, sr=None, mono=False, duration=None)
        # Convert to tensor
        audio_tensor = np.expand_dims(y, axis=0)
        audio_length = torch.tensor([audio_tensor.shape[-1]])
        ref_data.append([torch.from_numpy(audio_tensor), audio_length, sr])

    # Get B songs waveform characteristics
    print("\nLoading target group (B) & computing probabilities...")
    target_probs_list = []
    c = 0
    for s in song_list_B: 
        print('  {}/{}'.format(c, len(song_list_B)))
        c += 1
        # Load audio 
        y, sr = librosa.load(s, sr=None, mono=False, duration=None)
        # Convert to tensor
        audio_tensor = np.expand_dims(y, axis=0)
        audio_length = torch.tensor([audio_tensor.shape[-1]])
        data_tensor = torch.from_numpy(audio_tensor)
        target_probs = model._get_label_distribution(x=data_tensor, sizes=audio_length, sample_rates=sr)
        target_probs_list.append([data_tensor, audio_length, sr, target_probs])

    # Calculate KL div for A - B
    print("\nComputing KL divergence...") 
    for a in range(len(ref_data)): 
        print("{}/{}".format(a, len(song_list_A)))

        # Get reference probabilities from lable distribution of the PaSST classifier
        pred_probs = model._get_label_distribution(x=ref_data[a][0], sizes=ref_data[a][1], sample_rates=ref_data[a][2])
        for b in range(len(target_probs_list)): 
                print("  {}/{}".format(b, len(song_list_B)))
                
                # Get target probabilities from label distribution of the PaSST classifier
                target_probs = target_probs_list[b][3]
                
                # Compute KL divergence A-B, B-A and symmetric 
                kl_div_ab = kld_metric.kl_divergence(pred_probs, target_probs)
                kl_div_ba = kld_metric.kl_divergence(target_probs, pred_probs)
                kl_div_symm = (kl_div_ab+kl_div_ba)/2

                kldiv_ab.append(kl_div_ab.mean().item())
                kldiv_ba.append(kl_div_ba.mean().item())
                all_kldiv.append([song_list_A[a].split('/')[-1].split('.')[0], song_list_B[b].split('/')[-1].split('.')[0], kl_div_ab.mean().item(), kl_div_ba.mean().item(), kl_div_symm.mean().item()])
                

    df = pd.DataFrame(all_kldiv, columns = ['songA', 'songB', 'kl-div_AB', 'kl-div_BA', 'kl-div'])

    print('Results: ')
    print('Mean KLDiv-AB:', df['kl-div_AB'].mean())
    print('Mean KLDiv-BA:', df['kl-div_BA'].mean())
    print('Mean KLDiv:', df['kl-div'].mean())
    print('Median KLDiv:', df['kl-div'].median())

    if LOG_ACTIVE is True: 
        if logdir == 'log': 
            if not os.path.exists(logdir): os.makedirs(logdir)

        # independent file with results
        with open('{}/{}_kld.csv'.format(logdir, eval_name), 'w') as f: 
            writer = csv.writer(f)
            writer.writerow(['songA', 'songB', 'kl_div_ab', 'kl_div_ba', 'kl_div'])
            for r in range(len(all_kldiv)): 
                writer.writerow([all_kldiv[r][0], all_kldiv[r][1], all_kldiv[r][2], all_kldiv[r][3], all_kldiv[r][4]])

        file_path = f"{logdir}/{eval_name}_allresults.csv"
        # global file with all results 
        if os.path.exists(file_path): 
            print('\nFile exists: adding CLAP score results')
            data = pd.read_csv(file_path, dtype={'songA': str, 'songB': str})
            for r in all_kldiv: 
                mask = (data['songA'] == r[0]) & (data['songB'] == r[1])
                data.loc[mask, 'kl-div_AB'] = r[2]
                data.loc[mask, 'kl-div_BA'] = r[3]
                data.loc[mask, 'kl-div'] = r[4]

        else: 
            print('\nmissing file: creating new file with KL divergence score results')
            data = df
        
        data.to_csv('{}/{}_allresults.csv'.format(logdir, eval_name), index=False)



#############################################################################################################################################################

# Import information 
parser = argparse.ArgumentParser(description='No help available.')

# Input music 
    # Where are the songs located? 
    # groupA: reference group 
parser.add_argument('--a_samples', '-a', help='Indicate A samples directory.', required=True)

    # groupB: target group 
parser.add_argument('--b_samples', '-b', help='Indicate B samples directory.', required=True)

    # What was the code name assigned for this evaluation? 
parser.add_argument('--eval_name', help='Indicate eval name.', required=True)

# Do you want to register this results into the log folder?  
parser.add_argument('--log', help='Indicate if you do not want to register the results in the log folder or in which folder results should be stored.')

args = parser.parse_args()

folder_A = args.a_samples
folder_B = args.b_samples
eval_name = args.eval_name 
log = args.log

##############################################################################################################################################################

# Set time count
now = datetime.now()
# Run evaluation 
clap_eval(folder_A, folder_B, eval_name, log)

# Total time 
end = datetime.now()
time = end - now

print('\nElapsed time: ', time)
print('\n** Done! **\n')