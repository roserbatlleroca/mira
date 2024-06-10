#########################################
# MIRA TOOL: COVER ID ###################
#########################################

# Cover song identification distance 

# v1.0
# last update: june 2024

#########################################
# Import modules ########################
#########################################

import os
import csv
import numpy as np
#import matplotlib.pyplot as plt
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram
import glob
from scipy.io import wavfile
import argparse
from datetime import datetime
import pandas as pd

#######################################
# Similarity Evaluation ###############
#######################################

# Import information 
parser = argparse.ArgumentParser(description='No help available.')

# Set time count
now = datetime.now()

# Print computing status
print("\nCalculating CoverID...")

##############################################################################################################################################################

# Input music 
    # Where are the songs located? 
    # groupA 
parser.add_argument('--a_samples', '-a', help='Indicate A samples directory.', required=True)

    # groupB 
parser.add_argument('--b_samples', '-b', help='Indicate B samples directory.', required=True)

    # What was the code name assigned for this evaluation? 
parser.add_argument('--eval_name', help='Indicate eval name.', required=True)

# Do you want to register this results into the log folder?  
parser.add_argument('--log', help='Indicate if you do not want to register the results in the log folder.')

args = parser.parse_args()

folder_A = args.a_samples
folder_B = args.b_samples
eval_name = args.eval_name 
log = args.log

# Set if log is active by defining boolean LOG_ACTIVE
LOG_ACTIVE = False if log == 'no' else True

##############################################################################################################################################################

# Getting songs in the directory 
song_list_A = glob.glob(folder_A + '*.wav')
song_list_B = glob.glob(folder_B + '*.wav')

song_chromogram_A = []
song_chromogram_B = []
matrix_results = []

print("\nGetting samples chronograms...")
# Obtaining chromograms 
# Generated samples 
for s in range(len(song_list_A)): 
    # Get song_name
    song = song_list_A[s]
    song_loaded = estd.EasyLoader(filename=song_list_A[s], sampleRate=44100)()

    # Obtain chromogram 
    song_hpcp = hpcpgram(song_loaded, sampleRate=44100)
    song_chromogram_A.append((song.split('/')[-1], song_hpcp))

# Training samples 
for s in range(len(song_list_B)): 
    # Get song_name
    song = song_list_B[s]
    fs, data = wavfile.read(song_list_B[s])
    song_loaded = estd.EasyLoader(filename=song_list_B[s], sampleRate=44100)()

    # Obtain chromogram 
    song_hpcp = hpcpgram(song_loaded, sampleRate=44100)
    song_chromogram_B.append((song.split('/')[-1], song_hpcp))

print('\nCalculating Chroma Cross Similarity...')
# Get InterGroup Chroma Cross Similarity with cover song distance
# Generated samples vs Training samples
for a in range(len(song_chromogram_A)):  
    print("{}/{}".format(a, len(song_chromogram_A)))
    for b in range(len(song_chromogram_B)): 
        # Calculate Chross Croma Similarity 
        ccs = estd.ChromaCrossSimilarity()(song_chromogram_A[a][1], song_chromogram_B[b][1])
        # Obtainign Cover Song Distance 
        score_matrix, cs_distance = estd.CoverSongSimilarity(disOnset=0.5,
                                                disExtension=0.5,
                                                alignmentType='serra09',
                                                distanceType='asymmetric')(ccs)
        
        
        matrix_results.append((song_chromogram_A[a][0].split('.')[0], song_chromogram_B[b][0].split('.')[0], cs_distance))

df = pd.DataFrame(matrix_results, columns=['songA', 'songB', 'coverID'])

# Calulate mean and median distances 
coverid_mean = df['coverID'].mean()
coverid_median = df['coverID'].median()
print("Mean CoverID score:", coverid_mean)
print("Median CoverID score:", coverid_median)

if LOG_ACTIVE is True: 
    with open('log/{}_coverid.csv'.format(eval_name), 'w') as f: 
        writer = csv.writer(f)
        writer.writerow(['songA', 'songB', 'coverID'])
        for r in range(len(matrix_results)): 
            writer.writerow([matrix_results[r][0], matrix_results[r][1], matrix_results[r][2]])

    
     # global file with all results 
    if os.path.exists('log/{}_allresults.csv'.format(eval_name)): 
        print('file exists: adding CoverID score results')
        data = pd.read_csv('log/{}_allresults.csv'.format(eval_name)) 
        for r in matrix_results: 
            mask = (data['songA'] == r[0]) & (data['songB'] == r[1])
            data.loc[mask, 'coverID'] = r[2]
    else: 
        print('missing file: creating new file with CoverID score results')
        data = df
    
    data.to_csv('log/{}_allresults.csv'.format(eval_name), index=False)

end = datetime.now()
time = end - now

print('\nElapsed time: ', time)
print('\n** Done! **\n')