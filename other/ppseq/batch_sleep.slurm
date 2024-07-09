#!/bin/bash

#SBATCH --job-name=sleep_new			#name of the job to find when calling >>>sacct or >>>squeue
#SBATCH --nodes=1				#number of nodes, i.e. computers, to request off the cluster (nodes typically have ~20 singled threaded cores)
#SBATCH --ntasks=1				#how many independent script you are hoping to run 
#SBATCH --cpus-per-task=20			#how many threads to multithread across (no point more than number of cores available. also, you cannot thread across nodes) 
#SBATCH --time=250:00:00				#compute time
#SBATCH --mem=30gb				#memory to request
#SBATCH --output=run_out_animals_sleep.log		#where to save output log files (julia script prints here)  
#SBATCH --error=run_out_animals_sleep_new.err		#where to save output error files
#SBATCH --array=0-14         #task ID array for array scripting (can be passed to script below with command line argument --slurm-array-task-id $SLURM_ARRAY_TASK_ID)

pwd; hostname; date

export JULIA_NUM_THREADS=10

julia PPSeq_sleep_emmett.jl --data-directory "/ceph/sjones/projects/sequence_squad/organised_data/ppseq_data/prepared_data/striatum/Post_sleep/ppseq_ready/" --num-threads 20 --number-of-sequence-types 8  --results-directory "/ceph/sjones/projects/sequence_squad/organised_data/ppseq_data/output_data/striatum/New_Post_sleep/" --sacred-directory "/ceph/sjones/projects/sequence_squad/organised_data/ppseq_data/finalised_output/striatum/awake/" --slurm-array-task-id $SLURM_ARRAY_TASK_ID

sstat -j $SLURM_JOB_ID.batch --format=JobID,MaxVMSize,



