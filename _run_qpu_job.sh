#!/bin/bash
#PBS -N EN-CLSR
#PBS -l select=1:ncpus=1:ngpus=1:scratch_local=60gb:mem=60gb:gpu_mem=16gb
#PBS -l walltime=24:00:00



echo "Running EN-CLSR training from " $PBS_O_WORKDIR " on GPU.$CUDA_VISIBLE_DEVICES at cluster :: "`hostname`

cd $SCRATCHDIR || exit 1
cp $PBS_O_WORKDIR/* .
STARTTIME=$(date +%s)

trap 'cd $PBS_O_WORKDIR && qsub _run_asmd-auto.sh' TERM
trap 'clean_scratch' TERM EXIT
trap 'cp -r $SCRATCHDIR/*.out $SCRATCHDIR/* $PBS_O_WORKDIR && clean_scratch' TERM
## setup the enviroment ##

conda activate "/storage/brno2/home/mbryja/.conda/envs/PyTorch-gpu"

## run the clac ##
python3 main_incorporated_model.py


echo ""
ENDTIME=$(date +%s)
echo "scale=2; ($ENDTIME - $STARTTIME) / 60" | bc

cp -r ./* $PBS_O_WORKDIR


#Clean up Scratch
if [ -z "$SCRATCHDIR" ]; then echo "i am on local"; else rm -r *; fi