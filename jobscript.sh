#!/bin/ksh
#$ -q gpu
#$ -m abe
#$ -M sp165339@etu.u-bourgogne.fr
#$ -N segm_script_executor
#$ -o worklog.log

export PYTHONUSERBASE=/work/c-2iia/sp165339/venv
export PATH=/work/c-2iia/sp165339/venv/bin
export PYTHONIOENCODING=utf8
 
## job start time
## printf "\n Job started at : $(date)\n-----\n\n"
 
## what do we need? modules, envs etc.
module purge ## clear out the current module list
module load python/3.9.10
## source /work/c-2iia/sp165339/venv/bin/activate
## module purge ## clear out the current module list
## module load python/3.10
export PYTHONUSERBASE=/work/c-2iia/sp165339/TF_venv/
export PATH=/work/c-2iia/sp165339/TF_venv/bin
module load tensorflow/2.11.0/gpu
module list

## printf "\n --- begin of python execution ---\n"

## set a work dir OR variables with path to feed to python (ARGPARSER, click, etc)
cd /work/c-2iia/sp165339/Sickle_Cell_Segmentation_ResUNet_PP

## run script
python train.py --epochs 150


## printf "\n --- end of python execution ---\n\n"

## job end time
## printf "----\njob ended at $(date)\n\n"
