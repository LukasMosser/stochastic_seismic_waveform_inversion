#!/bin/sh
#PBS -l walltime=23:59:00
#PBS -l select=1:ncpus=32:ompthreads=32:mem=48000mb
#PBS -N waveform_test_revised
#PBS -m be
#PBS -J 0-100

module load gcc/8.2.0
module load mpi/intel-2018
module load cuda/9.0
module load cudnn/7.0
module load anaconda3/4.3.1

source activate devito2

export DEVITO_OPENMP="1";
export DEVITO_DLE="advanced"
export DEVITO_LOGGING="INFO"
export DEVITO_ARCH="gcc"

RESULTS_DIR="$PBS_O_WORKDIR/waveform/results_${sources}_sources_noise_25"

GENERATOR_PATH="$PBS_O_WORKDIR/waveform/checkpoints/generator_facies_multichannel_4_6790.pth"
DISCRIMINATOR_PATH="$PBS_O_WORKDIR/waveform/checkpoints/discriminator_facies_multichannel_4_6790.pth"
MINSMAXS_PATH="$PBS_O_WORKDIR/waveform/synthetics/half_circle_facies_vp_rho_mean_std_min_max.npy"
TESTIMGS_PATH="$PBS_O_WORKDIR/waveform/synthetics/test_half_circle_facies_vp_rho.npy"

CODE_WORK="$PBS_O_WORKDIR/waveform/code"
CODE_REMOTE="$TMPDIR/code"

mkdir "$CODE_REMOTE"

cp -r "$CODE_WORK" "$CODE_REMOTE"

cd "$CODE_REMOTE/code"


ls

nohup python main_paper_version_revised.py \
--num_runs 1 \
--sources ${sources} \
--print_to_console \
--generator_path $GENERATOR_PATH \
--discriminator_path $DISCRIMINATOR_PATH \
--minsmaxs_path $MINSMAXS_PATH \
--testimgs_path $TESTIMGS_PATH \
--working_dir $RESULTS_DIR \
--seed $PBS_ARRAY_INDEX \
--tensorboard \
--seismic_relative_error 0.5 \
--run_name "test_$PBS_ARRAY_INDEX" \
--noise_percent 0.25 \
