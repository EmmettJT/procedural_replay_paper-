## Program to run PPSeq on a section of data
# Other Imports
import DelimitedFiles: readdlm
import Random
import StatsBase: quantile
using CSV, DataFrames, Dates
import JSON


# INPUTS:
using ArgParse
s = ArgParseSettings()
@add_arg_table! s begin
    "--data-directory"
        default = "./data/preparedData/all_animals/" #where is data saved
    "--data-filename"
        default = "defaultData" #data name (don't add the ".txt")
    "--number-of-sequence-types"
        default = "6" # how many sequence types to try fit 
    "--PPSeq_file"
        default = "./PPSeq.jl/src/PPSeq.jl" #the PPSeq.jl file to use 
    "--num-threads"
        default = "1"
    "--results-directory"
        default = "./data/resultsData/all_animals/"
    "--results-name"
        default = nothing
    "--slurm-array-task-id"
        default = 0
    "--sacred-directory"
        default = nothing
    "--sacred-data-file"
        default = nothing
end

# Choose the right animal based on slurm array
parsed_args = parse_args(ARGS, s)

slurm_id = parse(Int64, parsed_args["slurm-array-task-id"])

# list_of_animals = ["136_1_2", "136_1_3", "136_1_4", "148_2_2", "149_1_1", "149_1_2", "149_1_3", "149_1_4", "149_2_1", "162_1_3", "178_1_1", "178_1_2", "178_1_3", "178_1_4", "178_1_5", "178_1_6", "178_1_7", "178_1_8", "178_1_9", "178_2_1", "178_2_2", "178_2_3", "178_2_4"]
# list_of_animals = ["255_1_1", "255_1_2","255_1_3", "255_1_4","256_1_1", "256_1_2","262_1_1", "262_1_2", "262_1_4", "262_1_5","262_1_6", "269_1_1","269_1_2","269_1_3","269_1_4","269_1_5","269_1_6","269_1_7", "270_1_1","270_1_2","270_1_3","270_1_4","270_1_5","270_1_6","270_1_7", "268_1_1","268_1_2","268_1_3","268_1_4","268_1_5","268_1_6","268_1_7","268_1_8","268_1_9","268_1_10"]
list_of_animals = ["262_1_7"]

parsed_args["data-filename"] = list_of_animals[slurm_id + 1]
parsed_args["results-name"] = string(list_of_animals[slurm_id + 1],"_run")

PPSeq_file = parsed_args["PPSeq_file"]

# Import PPSeq
include(PPSeq_file)
include("./save_results.jl")
const seq = PPSeq


Data_filename = string(parsed_args["data-directory"],parsed_args["data-filename"],".txt")
num_of_seq_types = parse(Int64, parsed_args["number-of-sequence-types"])
num_threads = parse(Int64,parsed_args["num-threads"])
if parsed_args["results-name"] == nothing
    parsed_args["results-name"] = parsed_args["data-filename"]
end 
#get number of neurons and maxtime from the data JSON file 
json_filename = string(parsed_args["data-directory"],"params_" , parsed_args["data-filename"],".json")
json_params = JSON.parsefile(json_filename)
num_neurons = json_params["number_of_neurons"]
# avg_firing_rate = json_params["average_firing_rate"]


timeslicelist = json_params["time_span"]
global max_time = 0
for timeslice in timeslicelist
    	global max_time
  	max_time = max_time + timeslice[2] - timeslice[1]
end
max_time = float(max_time)



# Load spikes.
spikes = seq.Spike[]
file_name = Data_filename
for (n, t) in eachrow(readdlm(file_name, '\t', Float64, '\n'))
    push!(spikes, seq.Spike(Int(n), t))
end

# Emmett edit:
# If there are some sacred sequences extract the relevant parameters here

if parsed_args["sacred-directory"] != nothing
    print("usign a sacred data file to fix some sequences")

    # List all files in the sacred-directory
    file_list = readdir(parsed_args["sacred-directory"])

    # Get the relevant file name that contains Current_Animal 
    for file in file_list
        if occursin(list_of_animals[slurm_id + 1], file)
	    print(" working - found a file")
            global sacred_data_file = parsed_args["sacred-directory"]*file*"/neuron_response.csv"
            break
        end
    end

    # Extract these parameters from the data files
    neuron_responses = CSV.read(sacred_data_file, DataFrame)

    sacred_sequences = ncol(neuron_responses)÷3
    print(typeof(sacred_sequences))

    if mod(nrow(neuron_responses), num_neurons) != 0
        error("Sacred Sequences have different number of neurons to input data!")
    end

    sacred_neuron_responses = neuron_responses[nrow(neuron_responses)-num_neurons+1:nrow(neuron_responses),:]
else
    sacred_sequences = 0
end


# Setup the config details
config = Dict(

    # Model hyperparameters
    :num_sequence_types =>  num_of_seq_types,
    :seq_type_conc_param => 1.0,
    :seq_event_rate => 1.2,

    :mean_event_amplitude => 23.44, #num_neurons*avg_firing_rate*0.3, # 100.0
    :var_event_amplitude => 23.44^2, #(num_neurons*avg_firing_rate*0.3)^2, # 1000.0

    :neuron_response_conc_param => 0.6,
    :neuron_offset_pseudo_obs => 0.5,
    :neuron_width_pseudo_obs => 1.0,
    :neuron_width_prior => 0.5,


    :num_warp_values => 1,
    :warp_type => 1,
    :max_warp => 30.0,
    :warp_variance => 1.0,

    :mean_bkgd_spike_rate => 70.32,
    :var_bkgd_spike_rate => 70.32,
    :bkgd_spikes_conc_param => 0.3,
    :max_sequence_length => 60.0,

    # MCMC Sampling parameters.
    :num_threads => num_threads,
    :num_anneals => 20,
    :samples_per_anneal => 100,
    :max_temperature => 40.0,
    :samples_after_anneal => 1000,
    :split_merge_moves_during_anneal => 10,
    :split_merge_moves_after_anneal => 10,
    :split_merge_window => 1.0,

    :save_every_after_anneal => 10,
    :save_every_during_anneal => 10,

    # Masking specific parameters
    :are_we_masking => 0, # Binary var, 1 = masking
    :mask_lengths => 5, # In seconds
    :percent_masked => 10, # percentage = number between 0 and 100 (not 0 and 1)
    :num_spike_resamples_per_anneal => 20, # How many resamplings of masked spikes per anneal
    :num_spike_resamples => 100, # How many times to resample masked spikes after annealing
    :samples_per_resample => 10, # How many times the unmasked spikes are sampled for each sampling

    # For training on some data and resampling on another set
    :sacred_sequences => sacred_sequences
);

#consistency check (multithreading cannot happen with masking)
if config[:are_we_masking] == 1
    #delete!(config,:num_threads)
    println("WARNING: MASKING PARAMETERS IS ON, YOU SURE YOU WANT OT MASK?")
    #print("Masking will not multithread")
end

# Then train the PPSeq Model
# Initialize all spikes to background process.
init_assignments = fill(-1, length(spikes))

# Construct model struct (PPSeq instance).
model = seq.construct_model(config, max_time, num_neurons)

# Run Gibbs sampling with an initial annealing period.
if config[:are_we_masking] == 1
    # First create a random set of masks

    config[:save_every_after_anneal] = min(config[:samples_per_resample], config[:save_every_after_anneal])
    config[:save_every_during_anneal] = min(config[:samples_per_resample], config[:save_every_during_anneal])

    masks = seq.create_random_mask(
        num_neurons,
        max_time,
        config[:mask_lengths] + 0.000001,
        config[:percent_masked]
    )

    masks = seq.clean_masks(masks, num_neurons)

    # Then run the easy sampler using these masks
    results = seq.easy_sample_masked!(model, spikes, masks, init_assignments, config)
    results_directory = parsed_args["results-directory"]*parsed_args["results-name"]*"_"*Dates.format(Dates.now(),"ddmmyyy_HHMM")*"/"
	masked_spikes, unmasked_spikes = seq.split_spikes_by_mask(spikes, masks)
    @time save_results_masked(results, config, results_directory, masked_spikes, unmasked_spikes)
    mkdir(results_directory*"trainingData/")
    cp(parsed_args["data-directory"]*parsed_args["data-filename"]*".txt", results_directory*"trainingData/"*parsed_args["data-filename"]*".txt")
    cp(parsed_args["data-directory"]*"params_"*parsed_args["data-filename"]*".json", results_directory*"trainingData/"*"params_"*parsed_args["data-filename"]*".json")

else
    # Check if there are any saved sequences that shouldn't be sampled
    if sacred_sequences != 0
        # Then make a new model that has these fixed parameters
        print("Sactifying model")
        model = seq.sanctify_model(model, Matrix(sacred_neuron_responses), config)
    end

    @time results = seq.easy_sample!(model, spikes, init_assignments, config);

    # Save the results
    # save_results(results, config, "../Simple"*Dates.format(Dates.now(),"HHMM"))
    results_directory = parsed_args["results-directory"]*parsed_args["results-name"]*"_"*Dates.format(Dates.now(),"ddmmyyy_HHMM")*"/"
    save_results(results, config, results_directory)
    mkdir(results_directory*"trainingData/")
    cp(parsed_args["data-directory"]*parsed_args["data-filename"]*".txt", results_directory*"trainingData/"*parsed_args["data-filename"]*".txt")
    cp(parsed_args["data-directory"]*"params_"*parsed_args["data-filename"]*".json", results_directory*"trainingData/"*"params_"*parsed_args["data-filename"]*".json")
end
