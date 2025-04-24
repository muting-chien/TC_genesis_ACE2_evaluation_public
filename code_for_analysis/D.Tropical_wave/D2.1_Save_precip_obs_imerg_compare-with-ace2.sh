#!/bin/bash

#SBATCH --partition=bar_all
#SBATCH --ntasks=16
#SBATCH --nodes=1 #1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-use=Mu-Ting.Chien@colostate.edu
#SBATCH --output=./txt/Save_precip_obs_ace2_imerg.txt

python Save_precip_obs_imerg_compare-with-ace2.py

