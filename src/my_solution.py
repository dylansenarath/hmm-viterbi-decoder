from pathlib import Path

# Resolve project root (…/hmm-viterbi-decoder) and the data folder
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

def data_path(name: str) -> Path:
    return DATA_DIR / name

#import time
#start_time = time.time()
# strip() saves the content of that line, removing all blank spaces from beginning and end of the string
# split() splits a string into a list where each word is a list item:
def parsing_weights(file_name):
    with open(file_name, 'r') as file:
        file_lines = file.readlines()

    data_indicator = file_lines[0].strip()
    entry_description = file_lines[1].split()
    num_entries = entry_description[0]
    if data_indicator == 'state_action_state_weights':
        num_unique_states = entry_description[1]
        num_unique_actions = entry_description[2]
        sas_default = entry_description[3]
    if data_indicator == 'state_observation_weights':
        num_unique_states = entry_description[1]
        num_unique_observations = entry_description[2]
        so_default = entry_description[3]

    weights = {}

    for line in file_lines[2:]:
        if line.strip():
            elements = line.strip().split()
            description = elements[:-1]
            cleaned_description = []
            for x in description:
                cleaned_description.append(x.replace('"', ''))
            if len(cleaned_description) > 1:
                description = tuple(cleaned_description)
            else:
                description = cleaned_description[0]
            weight = int(elements[-1])
            weights[description] = weight

    if data_indicator == 'state_weights':
        return weights, int(num_entries)
    if data_indicator == 'state_action_state_weights':
        return weights, int(num_entries), int(num_unique_states), int(num_unique_actions), int(sas_default)
    if data_indicator == 'state_observation_weights':
        return weights, int(num_entries), int(num_unique_states), int(num_unique_observations), int(so_default)


def normalize(states, actions, observations, transition_probabilities, emission_probabilities, default_sas, default_so):

    for previous_state in states:
        for action in actions:
            for current_state in states:
                if (previous_state, action, current_state) not in transition_probabilities:
                    transition_probabilities[(previous_state, action, current_state)] = default_sas
            sas_total = 0
            for current_state in states:
                sas_total += transition_probabilities[(previous_state, action, current_state)]
            for current_state in states:
                if sas_total != 0:
                    transition_probabilities[(previous_state, action, current_state)] /= sas_total
                else:
                    transition_probabilities[(previous_state, action, current_state)] = 0
    for state in states:
        for obs in observations:
            if (state, obs) not in emission_probabilities:
                emission_probabilities[(state, obs)] = default_so
        so_total = 0
        for obs in observations:
            so_total += emission_probabilities[state, obs]
        for obs in observations:
            if so_total != 0:
                emission_probabilities[(state, obs)] /= so_total
            else:
                emission_probabilities[(state, obs)] = 0

    return transition_probabilities, emission_probabilities

def normalize_initial_probs(weights):
    sum_of_weights = 0
    for x in weights:
        sum_of_weights += weights[x]

    normalized_weights = {}
    for description, weight in weights.items():
        normalized_weights[description] = int(weight) / sum_of_weights

    return normalized_weights


def viterbi_algorithm(obs, actions, states, start_probability, transition_probability, emission_probability):

    viterbi_probabilities = [{}]  # makes a probability table for the states. first element is time and second is state
    path = {}

    for state in states: # Set the probabilities for initial states based on observation
        viterbi_probabilities[0][state] = start_probability.get(state, 0) * emission_probability.get((state, obs[0]), 0)
        path[state] = [state]

    for time in range(1, len(obs)): # Calculate the probabilities for each state at every time interval
        viterbi_probabilities.append({})
        new_path = {}

        for state in states:
            max_prob = 0
            previous_state = None

            for previous in states:
                prob = viterbi_probabilities[time-1][previous] * transition_probability.get((previous, actions[time-1], state), 0) * emission_probability.get((state, obs[time]), 0)
                if prob > max_prob:
                    max_prob = prob
                    previous_state = previous
            viterbi_probabilities[time][state] = max_prob
            new_path[state] = path[previous_state] + [state]

        path = new_path

# Find which state has the highest probability at the final time and return the path leading to that state
    max_prob = 0
    final_state = None
    for state in states:
        prob = viterbi_probabilities[-1][state]
        if prob > max_prob:
            max_prob = prob
            final_state = state

    return path[final_state]


def write_output(path, state_sequence):
    with open(path, 'w') as output_file:
        output_file.write(f"states\n{len(state_sequence)}\n")
        for state in state_sequence:
            output_file.write(f'"{state}"\n')


# Create variable to hold the names of our input files
state_weights_file = data_path("state_weights.txt")
state_action_state_weights_file = data_path("state_action_state_weights.txt")
state_observation_weights_file = data_path("state_observation_weights.txt")
observation_actions_file = data_path("observation_actions.txt")

# Declare our output file which is just called states
output_file = "states.txt"

# Parse through the input weight files and extract key/value pairs to save as a dictionary list
# Identify if we have been given all pairs or if we need to use defaults
state_weights, num_states = parsing_weights(state_weights_file)
state_action_state_weights, num_sas_given, num_sas_unique_states, num_unique_actions, sas_default = parsing_weights(state_action_state_weights_file)
state_observation_weights, num_so_given, num_so_unique_states, num_unique_observations, so_default = parsing_weights(state_observation_weights_file)

# parse our observation_action file to separate into individual lists for actions and observations
with open(observation_actions_file, 'r') as obs_act_file:
    lines = obs_act_file.readlines()
    obs_actions = []
    for line in lines[2:]:
        clean_line = line.strip().split()
        for i in range(len(clean_line)):
            clean_line[i] = clean_line[i].replace('"', '')
        obs_actions.append(clean_line)
    observations = []
    for pair in obs_actions:
        observations.append(pair[0])
    actions = []
    for pair in obs_actions[:-1]:
        actions.append(pair[1])

possible_actions = []
states = list(state_weights.keys())
for sas in state_action_state_weights:
    act = sas[1]
    if act not in possible_actions:
        possible_actions.append(act)

possible_observations = []
for so in state_observation_weights:
    o = so[1]
    if o not in possible_observations:
        possible_observations.append(o)

# Normalize the probabilities
normalized_state_probability = normalize_initial_probs(state_weights)
normalized_state_action_state_probability, normalized_state_observation_probability = normalize(states, possible_actions, possible_observations, state_action_state_weights, state_observation_weights, sas_default, so_default)

predicted_states = viterbi_algorithm(observations, actions, states, normalized_state_probability, normalized_state_action_state_probability, normalized_state_observation_probability)

write_output(output_file, predicted_states)
