"""
Function to save the outputs of PPSeq into csv that we will then read in python
"""
function save_results(results, config, run_id)
   
    # ---------- anneal_latent_event_hist -----------
    latent_event_hist = results[:anneal_latent_event_hist]
    init_array = zeros(Float64, 0, 5)
    for (i, s) in enumerate(latent_event_hist)
        aux_array = zeros(Float64, length(s)+1, 5)
        for (j, z) in enumerate(s)
            aux_array[j, 1] = z.assignment_id
            aux_array[j, 2] = z.timestamp
            aux_array[j, 3] = z.seq_type
            aux_array[j, 4] = z.seq_warp
            aux_array[j, 5] = z.amplitude
        end
        aux_array[length(s)+1, :] = -1*ones(5)
        init_array = vcat(init_array, aux_array)
    end
    anneal_latent_event_hist_array = init_array
    ANNEAL_LATENT_EVENT_SHAPE = size(init_array)
   
    # ----------- latent_event_hist ------------
    latent_event_hist = results[:latent_event_hist]
    init_array = zeros(Float64, 0, 5)
    for (i, s) in enumerate(latent_event_hist)
        aux_array = zeros(Float64, length(s)+1, 5)
        for (j, z) in enumerate(s)
            aux_array[j, 1] = z.assignment_id
            aux_array[j, 2] = z.timestamp
            aux_array[j, 3] = z.seq_type
            aux_array[j, 4] = z.seq_warp
            aux_array[j, 5] = z.amplitude
        end
        aux_array[length(s)+1, :] = -1*ones(5)
        init_array = vcat(init_array, aux_array)
    end
    latent_event_hist_array = init_array
    LATENT_EVENT_SHAPE = size(init_array)
   
    latent_event_hist_frame = DataFrame(vcat(anneal_latent_event_hist_array, latent_event_hist_array),
        Symbol.(["assignment_id", "timestamp", "seq_type", "seq_warp", "amplitude"]))
   
    # --------- global_hist ----------
    global_vars_list = results[:globals_hist]
    N = size(global_vars_list[1].neuron_response_log_proportions)[1]
    R = size(global_vars_list[1].neuron_response_log_proportions)[2]
    init_array = zeros(Float64, 0, 3*R)
    seq_type_log_proportions = zeros(Float64, length(global_vars_list), R)
    bkgd_log_proportions = zeros(Float64, length(global_vars_list), N)
    for (i, s) in enumerate(global_vars_list)
        neuron_response = hcat(s.neuron_response_log_proportions, s.neuron_response_offsets,
        s.neuron_response_widths)
        init_array = vcat(init_array, neuron_response)
        seq_type_log_proportions[i, :] = s.seq_type_log_proportions
        bkgd_log_proportions[i, :] = s.bkgd_log_proportions
    end
    neuron_response_array = init_array
    seq_type_log_proportions_array = seq_type_log_proportions
    bkgd_log_proportions_array = bkgd_log_proportions
   
    # --------- anneal_global_hist ----------
        global_vars_list = results[:anneal_globals_hist]
    N = size(global_vars_list[1].neuron_response_log_proportions)[1]
    R = size(global_vars_list[1].neuron_response_log_proportions)[2]
    init_array = zeros(Float64, 0, 3*R)
    seq_type_log_proportions = zeros(Float64, length(global_vars_list), R)
    bkgd_log_proportions = zeros(Float64, length(global_vars_list), N)
    for (i, s) in enumerate(global_vars_list)
        neuron_response = hcat(s.neuron_response_log_proportions, s.neuron_response_offsets,
        s.neuron_response_widths)
        init_array = vcat(init_array, neuron_response)
        seq_type_log_proportions[i, :] = s.seq_type_log_proportions
        bkgd_log_proportions[i, :] = s.bkgd_log_proportions
    end
    anneal_neuron_response_array = init_array
    anneal_seq_type_log_proportions_array = seq_type_log_proportions
    anneal_bkgd_log_proportions_array = bkgd_log_proportions
   
    neuron_response_frame = DataFrame(vcat(anneal_neuron_response_array, neuron_response_array), :auto)
    seq_type_log_proportions_frame = DataFrame(vcat(anneal_seq_type_log_proportions_array, seq_type_log_proportions_array), :auto)
    bkgd_log_proportions_frame = DataFrame(vcat(anneal_bkgd_log_proportions_array, bkgd_log_proportions_array), :auto)
   
    # --------- assignment_hist ----------
    assigment_hist_frame = DataFrame(hcat(results[:anneal_assignment_hist], results[:assignment_hist]), :auto)
   
    aux_array = zeros(Float64, length(results[:log_p_hist]), 1)
    aux_array[:, 1] = results[:log_p_hist]
    log_p_hist_array = aux_array
   
    aux_array = zeros(Float64, length(results[:anneal_log_p_hist]), 1)
    aux_array[:, 1] = results[:anneal_log_p_hist]
    anneal_log_p_hist_array = aux_array
    log_p_hist_frame = DataFrame(vcat(anneal_log_p_hist_array, log_p_hist_array), :auto)
    # return log_p_hist_frame
   
    aux_array = zeros(Float64, length(results[:initial_assignments]), 1)
    aux_array[:, 1] = results[:initial_assignments]
    initial_assignments_frame = DataFrame(aux_array, :auto)
   
    results_directory = run_id
    if !isdir(results_directory)
        mkdir(results_directory)
    end

    #--------- input_data
   
    CSV.write(results_directory*"latent_event_hist.csv", latent_event_hist_frame, writeheader=true)
    CSV.write(results_directory*"neuron_response.csv", neuron_response_frame, writeheader=true)
    CSV.write(results_directory*"seq_type_log_proportions.csv", seq_type_log_proportions_frame, writeheader=true)
    CSV.write(results_directory*"bkgd_log_proportions_array.csv", bkgd_log_proportions_frame, writeheader=true)
    CSV.write(results_directory*"log_p_hist.csv", log_p_hist_frame, writeheader=true)
    CSV.write(results_directory*"initial_assignments.csv", initial_assignments_frame, writeheader=true)
    CSV.write(results_directory*"assigment_hist_frame.csv", assigment_hist_frame, writeheader=true)
   
    open(results_directory*"data_readme.txt", "w") do io
        write(io, "\n")
        write(io, "Important idea in the code that are translated in the results:\n")
	write(io, "\n")
        write(io, "Point process model that uses Gibbs sampling of an MCMC for the: \n")
        write(io, "1. Assignment spike to a cluster/sequence event ( with a part of given sequences sequence)\n")
	write(io, "2. The behaviour a sequence event \n")
	write(io, "3. The behaviour of each neuron in each sequence events (weight and spread of each neuron in each sequence)\n")
	write(io, "\n")
	write(io, "Therefore, we have a burning phase (anneal) and running phase before the MCMC posterior is accurately representing the given distributions \n")
	write(io, "We have sequence types and then in each sequence types we have sequence events we call a ‘sequence event/cluster’ \n")
	write(io, "\n")
	write(io, "1. Assignment spike to a cluster/sequence event \n")
	write(io, "\n")
        write(io, "Latent_events: [N_spikes, iter] \n")
        write(io, "anneal_latent_event (real_iter_anneal, n_sequence) : "*string(ANNEAL_LATENT_EVENT_SHAPE)*"\n")
        write(io, "after_anneal_latent_event (real_iter_after_anneal, n_sequence)  : "*string(LATENT_EVENT_SHAPE)*"\n")
        write(io, "full_latent_event: "*string(size(latent_event_hist_frame))*"\n")
        write(io, "Iterations separated by a row of -1 \n")
	write(io, "\n")
	write(io, "assigment_hist: n_spikes x (100 iters anneal + 200 iters) \n")
	write(io, "assigment_hist: "*string(size(assigment_hist_frame))*"\n")
	write(io, "The evolution of the assignment of each neurone to a  sequence event ( we write to file every 10 real iteration) \n")
	write(io, "\n")
	write(io, "initial_assignments(n_spikes x 1):"*string(size(initial_assignments_frame))*"\n")
	write(io, "Initial assignment of the neurone to the different sequence event or background.\n")
	write(io, "\n")
	write(io, "2. The behaviour a sequence event \n")
	write(io, "\n")
	write(io, "(We write to file every 10 real iteration) \n")
	write(io, "Latent_ event_hist.csv (nspikes,5) \n")
	write(io, "In this files you have sequence type, sequence event assignment number and time stamp for each neurones (Similarly for time wrap)\n")
	write(io, "\n")
	write(io, "3. The behaviour of each neuron in each sequence) \n")
        write(io, "\n")
        write(io, "Neuron response: [log_proportions, offsets, widths] (neurons*iters) × 3R(parameters neuron_response_log_proportions, s.neuron_response_offsets,
        s.neuron_response_widths) \n")
        write(io, "anneal_neuron_response(real_iter_anneal, n_sequence*3): "*string(size(anneal_neuron_response_array))*"\n")
        write(io, "after_neuron_response(real_iter,_after_anneal, n_sequence*3) : "*string(size(neuron_response_array))*"\n")
        write(io, "full_neuron_response: "*string(size(neuron_response_frame))*"\n")
	write(io, "Neuron response.csv "*"\n")
        write(io, "\n")
        write(io, "log_p_hist  (iters x 1) "*string(size(log_p_hist_frame))*", (iters x 1)\n")
 	write(io, "This is the logliklihood for all sequences event found in the data: \n")
	write(io, "\n")
        write(io, "bkgd_log_proportions (full_iter, neuron) : "*string(size(bkgd_log_proportions_frame))*"\n")
	write(io, "log likelihood of each neurone to belong to the background \n")
	write(io, "\n")
        write(io, "seq_type_log_proportions  (full_iter,sequence): "*string(size(seq_type_log_proportions_frame))*"\n")
	write(io, "Log likelihood of each sequence_type happening \n")
        write(io, "\n")
        write(io, "For more details on globals and events, see \n https://github.com/rodrigcd/PPSeq.jl/blob/84d8c1da3f7fe55f93555ae94242d1309c7d44fb/src/model/structs.jl")
    end

    jDict1 = JSON.json(config)
    f = open(results_directory*"config_file.json","w")
    JSON.print(f,jDict1)
    close(f)

    return latent_event_hist_frame
   
end






"""
MASKING VERSION
Use this function when you are using MASKING, saves a couple extra data files
Function to save the outputs of PPSeq into csv that we will then read in python
"""
function save_results_masked(results, config, run_id, masked_spikes, unmasked_spikes)

	# Masked and unmasked spike times and neuron id
	masked_spikes_dataframe = DataFrame(masked_spikes)
	unmasked_spikes_dataframe = DataFrame(unmasked_spikes)

    # ---------- anneal_latent_event_hist -----------
    latent_event_hist = results[:anneal_latent_event_hist]
    init_array = zeros(Float64, 0, 5)
    for (i, s) in enumerate(latent_event_hist)
        aux_array = zeros(Float64, length(s)+1, 5)
        for (j, z) in enumerate(s)
            aux_array[j, 1] = z.assignment_id
            aux_array[j, 2] = z.timestamp
            aux_array[j, 3] = z.seq_type
            aux_array[j, 4] = z.seq_warp
            aux_array[j, 5] = z.amplitude
        end
        aux_array[length(s)+1, :] = -1*ones(5)
        init_array = vcat(init_array, aux_array)
    end
    anneal_latent_event_hist_array = init_array
    ANNEAL_LATENT_EVENT_SHAPE = size(init_array)
   
    # ----------- latent_event_hist ------------
    latent_event_hist = results[:latent_event_hist]
    init_array = zeros(Float64, 0, 5)
    for (i, s) in enumerate(latent_event_hist)
        aux_array = zeros(Float64, length(s)+1, 5)
        for (j, z) in enumerate(s)
            aux_array[j, 1] = z.assignment_id
            aux_array[j, 2] = z.timestamp
            aux_array[j, 3] = z.seq_type
            aux_array[j, 4] = z.seq_warp
            aux_array[j, 5] = z.amplitude
        end
        aux_array[length(s)+1, :] = -1*ones(5)
        init_array = vcat(init_array, aux_array)
    end
    latent_event_hist_array = init_array
    LATENT_EVENT_SHAPE = size(init_array)
   
    latent_event_hist_frame = DataFrame(vcat(anneal_latent_event_hist_array, latent_event_hist_array),
        Symbol.(["assignment_id", "timestamp", "seq_type", "seq_warp", "amplitude"]))
   
    # --------- global_hist ----------
    global_vars_list = results[:globals_hist]
    N = size(global_vars_list[1].neuron_response_log_proportions)[1]
    R = size(global_vars_list[1].neuron_response_log_proportions)[2]
    init_array = zeros(Float64, 0, 3*R)
    seq_type_log_proportions = zeros(Float64, length(global_vars_list), R)
    bkgd_log_proportions = zeros(Float64, length(global_vars_list), N)
    for (i, s) in enumerate(global_vars_list)
        neuron_response = hcat(s.neuron_response_log_proportions, s.neuron_response_offsets,
        s.neuron_response_widths)
        init_array = vcat(init_array, neuron_response)
        seq_type_log_proportions[i, :] = s.seq_type_log_proportions
        bkgd_log_proportions[i, :] = s.bkgd_log_proportions
    end
    neuron_response_array = init_array
    seq_type_log_proportions_array = seq_type_log_proportions
    bkgd_log_proportions_array = bkgd_log_proportions
   
    # --------- anneal_global_hist ----------
    global_vars_list = results[:anneal_globals_hist]
    N = size(global_vars_list[1].neuron_response_log_proportions)[1]
    R = size(global_vars_list[1].neuron_response_log_proportions)[2]
    init_array = zeros(Float64, 0, 3*R)
    seq_type_log_proportions = zeros(Float64, length(global_vars_list), R)
    bkgd_log_proportions = zeros(Float64, length(global_vars_list), N)
    for (i, s) in enumerate(global_vars_list)
        neuron_response = hcat(s.neuron_response_log_proportions, s.neuron_response_offsets,
        s.neuron_response_widths)
        init_array = vcat(init_array, neuron_response)
        seq_type_log_proportions[i, :] = s.seq_type_log_proportions
        bkgd_log_proportions[i, :] = s.bkgd_log_proportions
    end
    anneal_neuron_response_array = init_array
    anneal_seq_type_log_proportions_array = seq_type_log_proportions
    anneal_bkgd_log_proportions_array = bkgd_log_proportions
   
    neuron_response_frame = DataFrame(vcat(anneal_neuron_response_array, neuron_response_array), :auto)
    seq_type_log_proportions_frame = DataFrame(vcat(anneal_seq_type_log_proportions_array, seq_type_log_proportions_array), :auto)
    bkgd_log_proportions_frame = DataFrame(vcat(anneal_bkgd_log_proportions_array, bkgd_log_proportions_array), :auto)
   
    # --------- assignment_hist ----------
    assigment_hist_frame = DataFrame(hcat(results[:anneal_assignment_hist], results[:assignment_hist]), :auto)

   # ---------- train log hist ----------
    aux_array = zeros(Float64, length(results[:train_log_p_hist]), 1)
    aux_array[:, 1] = results[:train_log_p_hist]
    train_log_p_hist_array = aux_array
   
    aux_array = zeros(Float64, length(results[:anneal_train_log_p_hist]), 1)
    aux_array[:, 1] = results[:anneal_train_log_p_hist]
    anneal_train_log_p_hist_array = aux_array
    train_log_p_hist_frame = DataFrame(vcat(anneal_train_log_p_hist_array, train_log_p_hist_array), :auto)
    # return log_p_hist_frame

   # ---------- test log hist ----------
    aux_array = zeros(Float64, length(results[:test_log_p_hist]), 1)
    aux_array[:, 1] = results[:test_log_p_hist]
    test_log_p_hist_array = aux_array

    aux_array = zeros(Float64, length(results[:anneal_test_log_p_hist]), 1)
    aux_array[:, 1] = results[:anneal_test_log_p_hist]
    anneal_test_log_p_hist_array = aux_array
    test_log_p_hist_frame = DataFrame(vcat(anneal_test_log_p_hist_array, test_log_p_hist_array), :auto)
    # return log_p_hist_frame


    aux_array = zeros(Float64, length(results[:initial_assignments]), 1)
    aux_array[:, 1] = results[:initial_assignments]
    initial_assignments_frame = DataFrame(aux_array, :auto)
   
    results_directory = run_id
    # *"_results/"
    if !isdir(results_directory)
        mkdir(results_directory)
    end
   
    CSV.write(results_directory*"latent_event_hist.csv", latent_event_hist_frame, writeheader=true)
    CSV.write(results_directory*"neuron_response.csv", neuron_response_frame, writeheader=true)
    CSV.write(results_directory*"seq_type_log_proportions.csv", seq_type_log_proportions_frame, writeheader=true)
    CSV.write(results_directory*"bkgd_log_proportions_array.csv", bkgd_log_proportions_frame, writeheader=true)
    CSV.write(results_directory*"train_log_p_hist.csv", train_log_p_hist_frame, writeheader=true)
    CSV.write(results_directory*"test_log_p_hist.csv", test_log_p_hist_frame, writeheader=true)
    CSV.write(results_directory*"initial_assignments.csv", initial_assignments_frame, writeheader=true)
    CSV.write(results_directory*"assigment_hist_frame.csv", assigment_hist_frame, writeheader=true)
	CSV.write(results_directory*"masked_spikes.csv", masked_spikes_dataframe, writeheader=true)
   	CSV.write(results_directory*"unmasked_spikes.csv", unmasked_spikes_dataframe, writeheader=true)

    open(results_directory*"data_readme.txt", "w") do io
        write(io, "\n")
        write(io, "Latent_events: [N_spikes, iter] \n")
        write(io, "anneal_latent_event: "*string(ANNEAL_LATENT_EVENT_SHAPE)*"\n")
        write(io, "after_anneal_latent_event: "*string(LATENT_EVENT_SHAPE)*"\n")
        write(io, "full_latent_event: "*string(size(latent_event_hist_frame))*"\n")
        write(io, "Iterations separated by a row of -1 \n")
        write(io, "\n")
        write(io, "Neuron response: [log_proportions, offsets, widths] (neurons*iters) × 3R \n")
        write(io, "anneal_neuron_response: "*string(size(anneal_neuron_response_array))*"\n")
        write(io, "after_neuron_response: "*string(size(neuron_response_array))*"\n")
        write(io, "full_neuron_response: "*string(size(neuron_response_frame))*"\n")
        write(io, "\n")
        write(io, "assigment_hist: n_spikes x (100 iters anneal + 200 iters) \n")
        write(io, "assigment_hist: "*string(size(assigment_hist_frame))*"\n")
        write(io, "train_log_p_hist: "*string(size(train_log_p_hist_frame))*", (iters x 1)\n")
        write(io, "test_log_p_hist: "*string(size(test_log_p_hist_frame))*", (iters x 1)\n")
        write(io, "bkgd_log_proportions: "*string(size(bkgd_log_proportions_frame))*"\n")
        write(io, "seq_type_log_proportions: "*string(size(seq_type_log_proportions_frame))*"\n")
        write(io, "(anneal iters) + (without anneal iters) \n")
        write(io, "\n")
        write(io, "initial_assignments, "*string(size(initial_assignments_frame))*"\n")
        write(io, "For more details on globals and events, see \n https://github.com/rodrigcd/PPSeq.jl/blob/84d8c1da3f7fe55f93555ae94242d1309c7d44fb/src/model/structs.jl")
    end

    jDict1 = JSON.json(config)
    f = open(results_directory*"config_file.json","w")
    JSON.print(f,jDict1)
    close(f)

    return latent_event_hist_frame
   
end
