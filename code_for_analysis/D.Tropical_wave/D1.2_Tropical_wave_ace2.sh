#!/bin/bash

#SBATCH --partition=bar_all
#SBATCH --ntasks=16
#SBATCH --nodes=1 #1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-use=Mu-Ting.Chien@colostate.edu
##SBATCH --output=../txt/Tropical_wave.txt
#SBATCH --output=../txt/Tropical_wave_ace2.txt
##SBATCH --output=../txt/Tropical_wave_obs.txt
##SBATCH --output=../txt/Tropical_wave_filtering.txt

#python Tropical_wave.py
python Tropical_wave_ace2.py
#python Tropical_wave_obs.py
#python Tropical_wave_filtering.py