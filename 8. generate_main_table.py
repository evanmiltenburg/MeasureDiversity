"""
This file can only be run using Python 3, because it uses f-strings to format the
table caption.
"""

import json
from methods import load_json
from tabulate import tabulate

def load_system_stats(name):
    "Load system stats based on the system name."
    base = './Data/Systems/'
    path = base + name + '/Val/stats.json'
    return load_json(path)


def format_vals(stats):
    "Function to format the numbers in the stats."
    stats['average_sentence_length'] = "{:.1f}".format(stats['average_sentence_length'])
    stats['std_sentence_length'] = "{:.2f}".format(stats['std_sentence_length'])
    stats['type_token_ratio'] = "{:.2f}".format(stats['type_token_ratio'])
    stats['bittr'] = "{:.2f}".format(stats['bittr'])
    stats['ttr10k'] = "{:.2f}".format(stats['ttr10k'])
    stats['ttr100k'] = "{:.2f}".format(stats['ttr100k'])
    if 'percentage_novel' in stats:
        stats['percentage_novel'] = "{:.1f}".format(stats['percentage_novel'])


def get_values(d, keys, to_print=True):
    "Get the corresponding values for a list of keys."
    if to_print:
        format_vals(d)
    return [d[key] for key in keys]


def get_system_row(system_name, systems, keys):
    "Get table row for a system."
    data = systems[system_name]
    return get_values(data, keys)


systems = {'Dai-et-al-2017': "\paperdata{Dai_2017_ICCV}{GAN with policy gradient}",
           'Liu-et-al-2017': "\paperdata{liu2017mat}{Object detector+LSTM}",
           'Mun-et-al-2017': "\paperdata{mun2017text}{Text-guided attention LSTM}",
           'Shetty-et-al-2016': '\paperdata{Shetty:2016:ESC:2983563.2983571}{CNN\&scene features+LSTM}',
           'Shetty-et-al-2017': '\paperdata{Shetty_2017_ICCV}{GAN with Gumbel softmax}',
           'Tavakoli-et-al-2017': '\paperdata{tavakoli2017paying}{CNN\&saliency features+LSTM}',
           'Vinyals-et-al-2017': '\paperdata{vinyals2017show}{CNN+LSTM}',
           'Wu-et-al-2016': '\paperdata{wu2017image}{Attribute vector+LSTM}',
           'Zhou-et-al-2017': '\paperdata{zhou2017watch}{CNN+text-conditional LSTM}'}

train_stats  = load_json('./Data/COCO/Processed/train_stats.json')
val_stats    = load_json('./Data/COCO/Processed/val_stats.json')
system_stats = {sys_name: load_system_stats(sys_name) for sys_name in systems}
bleu_meteor  = load_json('./Data/Systems/bleu_meteor.json')
global_recall = load_json('./Data/Output/global_recall.json')

headers     = ['Name', 'Ref', 'Technique', 'BLEU', 'Meteor', "ASL", "SDSL", "Types", "MSTTR", 'BiTTR', '\%Novel', 'Coverage']
system_keys = ["average_sentence_length", 'std_sentence_length', "num_types", "type_token_ratio", 'bittr', 'percentage_novel']
corpus_keys = ["average_sentence_length", 'std_sentence_length', "avg_types", "type_token_ratio", 'bittr', 'percentage_novel']
train_keys  = corpus_keys[:-1] + ["total_types", "total_tokens"]

train_results = list(zip(train_keys, get_values(train_stats, train_keys)))

rows = []
for system in systems:
    lead = [system, '', '']
    reported_scores = [bleu_meteor[system]['BLEU'], bleu_meteor[system]['Meteor']]
    general_metrics = get_system_row(system, system_stats, system_keys)
    global_recall_score = ['{:.2f}'.format(global_recall[system]['score'])]
    row = lead + reported_scores + general_metrics + global_recall_score
    rows.append(row)

val_row = ['Val','','Reference corpus', '--', '--'] + get_values(val_stats, corpus_keys) + ['--']
rows.append(val_row)

main_table = tabulate(rows, headers, tablefmt="latex_booktabs")

def modify_table(main_table):
    "Super specific function to properly add citations."
    special_command = "\\newcommand{\paperdata}[2]{\citeauthor{#1} (\citeyear{#1}) & \cite{#1} & #2}\n\n"
    main_table = special_command + main_table
    val_stuff = "&       & Reference corpus &"
    main_table = main_table.replace(val_stuff, "BUNNY")   # Make placeholder.
    main_table = main_table.replace("&       &", '')      # Remove these columns.
    for system, latex in systems.items():
        main_table = main_table.replace(system, latex)
    main_table = main_table.replace('BUNNY', val_stuff)   # Replace placeholder.
    return main_table

main_table = modify_table(main_table)

caption = f"""The number of Types and Tokens, Average Sentence Length (ASL), and
normalized Type-Token Ratio (nTTR, number of types per 1K tokens). Note that the
results for the validation set are averaged over the parallel descriptions. The total
number of types in the validation set is {val_stats['total_types']}, and the total
number of tokens is {val_stats['total_tokens']}. The ASL and nTTR for the training
set are similar to the validation set."""

print(main_table)
print("Caption:", caption,'\n\n')
print("------------")
print("Train stats:")
for key,val in train_results:
    print(key, '\t', val)
print("------------")

with open('./Data/Output/main_table.txt','w') as f:
    f.write(main_table)
    f.write('\n')
    f.write('Caption: '+ caption)
    f.write("\n------------\nTrain stats:\n")
    for key,val in train_results:
        f.write(f'{key}\t\t{val}\n')
    f.write("------------")
    
