import pandas as pd
import numpy as np
import os
import json


class DataLoader(object):
    def __init__(self, data_path, spike_path, transition_data_path=None):
        self.spikes = pd.read_csv(spike_path, delimiter="\t", header=None)
        if not transition_data_path is None:
            self.transition_data = pd.read_csv(transition_data_path)
        self.data_path = data_path
        self.file_list = [f for f in os.listdir(self.data_path)]
        self._load_csvs()

    def show_readme(self):
        with open(self.readme_file_path) as f:
            for line in f:
                print(line)

    def _load_csvs(self):
        for f in self.file_list:
            try:
                name, file_format = f.split(".")
            except: 
                continue

            if file_format == "csv":
                file_path = os.path.join(self.data_path, f)
                frame = pd.read_csv(file_path)
                setattr(self, name, frame)
            elif name == "data_readme":
                self.readme_file_path = os.path.join(self.data_path, f)
            elif file_format == "json":
                with open(os.path.join(self.data_path, f)) as json_file:
                    self.config = json.load(json_file)
                    self.config = json.loads(self.config)


def sortperm_neurons(data, sequence_ordering=None, th=0.2):
    N_neurons = data.bkgd_log_proportions_array.shape[1]
    n_sequences = data.config["num_sequence_types"]
    all_final_globals = data.neuron_response.iloc[-N_neurons:]
    resp_prop = np.exp(all_final_globals.values[:, :n_sequences])
    offset = all_final_globals.values[-N_neurons:, n_sequences:2*n_sequences]
    peak_response = np.amax(resp_prop, axis=1)
    has_response = peak_response > np.quantile(peak_response, th)
    preferred_type = np.argmax(resp_prop, axis=1)
    if sequence_ordering == None:
        ordered_preferred_type = preferred_type
    else:
        ordered_preferred_type = np.zeros(N_neurons)
        for seq in range(n_sequences):
            seq_indices = np.where(preferred_type == sequence_ordering[seq])
            ordered_preferred_type[seq_indices] = seq
    preferred_delay = offset[np.arange(N_neurons), preferred_type]
    Z = np.stack([has_response, ordered_preferred_type+1, preferred_delay], axis=1)
    indexes = np.lexsort((Z[:, 2], Z[:, 1], Z[:, 0]))
    return indexes

if __name__ == "__main__":
    path = "./results/n_seq3_results"
    data = DataLoader(data_path=path, spike_path="./data/testing_n_sequences.txt")
    print(data.file_list)
    print(data.bkgd_log_proportions_array)
    data.show_readme()
    print(data.neuron_response)

    sortperm_neurons(data)

    print(data.config)
