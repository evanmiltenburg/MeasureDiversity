from methods import load_json
from nltk.corpus import wordnet as wn
from collections import Counter, defaultdict
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context('paper', font_scale=7)
sns.set_palette(sns.color_palette("cubehelix", 10))

#################################################################################
# Main functions

def average_depth(word, pos):
    "Compute average depth for all synsets corresponding to a word."
    synsets = wn.synsets(word, pos)
    depths = [s.min_depth() for s in synsets]
    avg_depth = np.average(depths)
    return avg_depth


def depth_stats(entries):
    # Create a list of all noun tokens.
    noun_tokens = [word.lower() for entry in entries
                        for word, pos in entry['tagged']
                        if pos.startswith('NN')]
    # Create a list of all noun types.
    noun_types        = set(noun_tokens)
    # Create a dictionary of all average synset depths for each word.
    depths            = {word: average_depth(word, 'n') for word in noun_types}
    # Use the dictionary to create a list of all depths.
    depths_per_type  = [depths[word] for word in noun_types]
    depths_per_token = [depths[word] for word in noun_tokens]
    # Filter weights
    depths_per_type  = [w for w in depths_per_type if not np.isnan(w)]
    depths_per_token = [w for w in depths_per_token if not np.isnan(w)]
    # Return stats.
    return dict(average_type_depth = np.average(depths_per_type),
                average_token_depth = np.average(depths_per_token),
                std_type_depth = np.std(depths_per_type),
                std_token_depth = np.std(depths_per_token))


def depth_including_compounds(entries):
    noun_tokens = nouns_from_entries(entries)
    # Create a list of all noun types.
    noun_types        = set(noun_tokens)
    # Create a dictionary of all average synset depths for each word.
    depths            = {word: average_depth(word, 'n') for word in noun_types}
    # Use the dictionary to create a list of all depths.
    depths_per_type  = [depths[word] for word in noun_types]
    depths_per_token = [depths[word] for word in noun_tokens]
    # Filter weights
    depths_per_type  = [w for w in depths_per_type if not np.isnan(w)]
    depths_per_token = [w for w in depths_per_token if not np.isnan(w)]
    # Return stats.
    return dict(average_type_depth = np.average(depths_per_type),
                average_token_depth = np.average(depths_per_token),
                std_type_depth = np.std(depths_per_type),
                std_token_depth = np.std(depths_per_token))


def nouns_from_entries(entries):
    "Return a list of compounds from the document."
    nouns = []
    current = []
    for entry in entries:
        for token, pos in entry['tagged']:
            if pos.startswith('NN'):
                current.append(token.lower())
            elif len(current) >= 1:
                joined = '_'.join(current) # wn.synsets('fire_engine','n') works :)
                nouns.append(joined)
                current = []
        if len(current) >= 1:
            joined = '_'.join(current)
            nouns.append(joined)
            current = []
    return nouns

def get_depths_histogram(entries):
    noun_tokens = nouns_from_entries(entries)
    # Create a list of all noun types.
    noun_types        = set(noun_tokens)
    # Create a dictionary of all average synset depths for each word.
    depths            = {word: average_depth(word, 'n') for word in noun_types}
    # Use the dictionary to create a list of all depths.
    depths_per_type  = [depths[word] for word in noun_types]
    depths_per_token = [depths[word] for word in noun_tokens]
    # Filter weights
    depths_per_type  = [w for w in depths_per_type if not np.isnan(w)]
    depths_per_token = [w for w in depths_per_token if not np.isnan(w)]
    # Compute histograms
    return dict(type_histogram = Counter(map(round, depths_per_type)),
                token_histogram = Counter(map(round, depths_per_token)))


def plot_relative(systems, val, kind='token_histogram'):
    fig, ax = plt.subplots(figsize=(30,20))
    lw = 8.0
    ms = 25.0
    keys = set()
    keys.update(val[kind].keys())
    for name, data in systems.items():
        keys.update(data[kind].keys())
    keys = sorted(keys)
    
    for name, data in systems.items():
        total = sum(data[kind].values())
        y = [data[kind][key]/total for key in keys]
        plt.plot(keys, y, 'o-', label=name, linewidth=lw, markersize=ms)
    
    total = sum(val[kind].values())
    y = [val[kind][key]/total for key in keys]
    plt.plot(keys, y, 'o-', label='Val', linewidth=lw, markersize=ms)
    
    plt.legend()#loc=2, bbox_to_anchor=(0, 1.1))
    plt.xticks(keys)
    plt.ylabel('Proportion')
    plt.xlabel('WordNet depth')
    sns.despine()
    plt.savefig(f'./Data/Output/wordnet_relative_{kind}.pdf')


def plot_absolute(systems, val, kind='token_histogram'):
    fig, ax = plt.subplots(figsize=(30,20))
    lw = 8.0
    ms = 25.0
    keys = set()
    keys.update(val[kind].keys())
    for name, data in systems.items():
        keys.update(data[kind].keys())
    keys = sorted(keys)
    
    for name, data in systems.items():
        y = [data[kind][key] for key in keys]
        plt.plot(keys, y, 'o-', label=name, linewidth=lw, markersize=ms)
    
    y = [val[kind][key] for key in keys]
    plt.plot(keys, y, 'o-', label='Val', linewidth=lw, markersize=ms)
    
    plt.legend()#loc=2, bbox_to_anchor=(0, 1.1))
    plt.xticks(keys)
    plt.ylabel('Number of tokens' if kind=='token_histogram' else 'Number of types')
    plt.xlabel('WordNet depth')
    sns.despine()
    plt.savefig(f'./Data/Output/wordnet_absolute_{kind}.pdf')

################################################################################
# Helpers

def parallel_entries(val_tagged):
    "Get parallel lists of entries."
    d = defaultdict(list)
    for entry in val_tagged['annotations']:
        img_id = entry['image_id']
        d[img_id].append(entry)
    return list(zip(*d.values()))

def average_dicts(dict_list):
    main_dict = defaultdict(list)
    for d in dict_list:
        for k,v in d.items():
            main_dict[k].append(v)
    for k,v in main_dict.items():
        main_dict[k] = float(sum(v))/len(v)
    return main_dict

def load_system_data(name):
    base = './Data/Systems/'
    path = base + name + '/Val/annotated.json'
    return load_json(path)

def get_keys(d, keys):
    return [d[key] for key in keys]

################################################################################
# Compute stats.

###########################
# Val

val_tagged       = load_json('./Data/COCO/Processed/tagged_val2014.json')
parallel_entries = parallel_entries(val_tagged)
parallel_results = [depth_including_compounds(entries) for entries in parallel_entries]
val_result       = average_dicts(parallel_results)

parallel_histos  = [get_depths_histogram(entries) for entries in parallel_entries]
type_histos = [d['type_histogram'] for d in parallel_histos]
token_histos = [d['token_histogram'] for d in parallel_histos]
val_histo = dict(type_histogram=average_dicts(type_histos),
                 token_histogram=average_dicts(token_histos))

###########################
# Systems

systems = {'Dai-et-al-2017': "\citeauthor{Dai_2017_ICCV} (\citeyear{Dai_2017_ICCV})",
           'Liu-et-al-2017': "\citeauthor{liu2017mat} (\citeyear{liu2017mat})",
           'Mun-et-al-2017': "\citeauthor{mun2017text} (\citeyear{mun2017text})",
           'Shetty-et-al-2016': '\citeauthor{Shetty:2016:ESC:2983563.2983571} (\citeyear{Shetty:2016:ESC:2983563.2983571})',
           'Shetty-et-al-2017': '\citeauthor{Shetty_2017_ICCV} (\citeyear{Shetty_2017_ICCV})',
           'Tavakoli-et-al-2017': '\citeauthor{tavakoli2017paying} (\citeyear{tavakoli2017paying})',
           'Vinyals-et-al-2017': '\citeauthor{vinyals2017show} (\citeyear{vinyals2017show})',
           'Wu-et-al-2016': '\citeauthor{wu2017image} (\citeyear{wu2017image})',
           'Zhou-et-al-2017': '\citeauthor{zhou2017watch} (\citeyear{zhou2017watch})'}

loaded_systems = {system: load_system_data(system) for system in systems}
system_results = {system: depth_including_compounds(loaded_data) for system, loaded_data in loaded_systems.items()}
system_histos = {system: get_depths_histogram(loaded_data) for system, loaded_data in loaded_systems.items()}

###########################
# Table

keys = ['average_type_depth', 'std_type_depth', 'average_token_depth', 'std_token_depth']
headers = ['Name', 'Avg', 'Std', 'Avg', 'Std']
rows = []
for system, results in system_results.items():
    row = [system] + get_keys(results, keys)
    rows.append(row)

val_row = ['Val'] + get_keys(val_result, keys)
rows.append(val_row)
table = tabulate(rows, headers=headers, tablefmt='latex_booktabs', floatfmt=".2f")
for system, cite in systems.items():
    table = table.replace(system, cite)

table = table.replace('\\toprule', '\\toprule \n & \multicolumn{2}{c}{Types} & \multicolumn{2}{c}{Tokens}\\\\\n \cmidrule(lr){2-3} \cmidrule(lr){4-5}')
print(table)

with open('./Data/Output/wordnet_depths_table.txt', 'w') as f:
    f.write(table)

###########################
# Plot

#plot_absolute(system_histos, val_histo)
plot_relative(system_histos, val_histo, kind='token_histogram')
plot_relative(system_histos, val_histo, kind='type_histogram')
plot_absolute(system_histos, val_histo, kind='token_histogram')
plot_absolute(system_histos, val_histo, kind='type_histogram')
