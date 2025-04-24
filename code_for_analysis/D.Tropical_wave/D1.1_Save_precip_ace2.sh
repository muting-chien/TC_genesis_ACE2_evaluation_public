#!/bin/bash

#SBATCH --partition=bar_all
#SBATCH --ntasks=8
#SBATCH --nodes=1 #1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-use=Mu-Ting.Chien@colostate.edu
#SBATCH --output=./txt/Save_precip_ace2.txt

python Save_precip_ace2.py

