#!/bin/bash

#SBATCH --job-name=awake		#name of the job to find when calling >>>sacct or >>>squeue
#SBATCH --nodes=1				#number of nodes, i.e. computers, to request off the cluster (nodes typically have ~20 singled threaded cores)
#SBATCH --ntasks=1				#how many independent script you are hoping to run 
#SBATCH --cpus-per-task=20			#how many threads to multithread across (no point more than number of cores available. also, you cannot thread across nodes) 
#SBATCH --time=80:00:00				#compute time
#SBATCH --mem=16gb				#memory to request
#SBATCH --output=run_out_animals.log		#where to save output log files (julia script prints here)  
#SBATCH --error=run_out_animals.err		#where to save output error files
#SBATCH --array=0-1            #task ID array for array scripting (can be passed to script below with command line argument --slurm-array-task-id $SLURM_ARRAY_TASK_ID)

pwd; hostname; date

export JULIA_NUM_THREADS=10

julia PPSeq_awake_emmett.jl --data-directory "/ceph/sjones/projects/sequence_squad/organised_data/ppseq_data/prepared_data/striatum/Awake/ppseq_ready/" --num-threads 10 --results-directory "/ceph/sjones/projects/sequence_squad/organised_data/ppseq_data/output_data/striatum/awake/New_Awake" --slurm-array-task-id $SLURM_ARRAY_TASK_ID

sstat -j $SLURM_JOB_ID.batch --format=JobID,MaxVMSize,

