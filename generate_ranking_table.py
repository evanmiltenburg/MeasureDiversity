from methods import load_json
from collections import Counter
from string import punctuation
from tabulate import tabulate

def name_to_stats_path(name):
    "Get mapping based on system name."
    base = './Data/Systems/'
    path = base + name + '/Val/stats.json'
    return load_json(path)


def list_from_counts(count_tuples):
    "Get list of words from list of count tuples."
    print(count_tuples[:ranking_length])
    return [w for w,c in count_tuples]

################################################################################
# Parameters.

ranking_length=20

systems = ['Dai-et-al-2017',
           'Liu-et-al-2017',
           'Mun-et-al-2017',
           'Shetty-et-al-2016',
           'Shetty-et-al-2017',
           'Tavakoli-et-al-2017',
           'Vinyals-et-al-2017',
           'Wu-et-al-2016',
           'Zhou-et-al-2017']

################################################################################
# Global ranking.


def get_top_n_omitted(stats, not_learned, n=10):
    "Get the top-n omitted words."
    omissions = {word: count for word, count in stats['total_counts'].items()
                             if word in not_learned}
    
    # Convert to counter.
    omissions = Counter(omissions)
    
    # Clean the data.
    del omissions['..']
    for char in punctuation + ' \n':
        del omissions[char]
    
    # Return most common omissions.
    top_n = omissions.most_common(n)
    return list_from_counts(top_n)


train_stats = load_json('./Data/COCO/Processed/train_stats.json')
val_stats = load_json('./Data/COCO/Processed/val_stats.json')

train     = set(train_stats['types'])
val       = set(val_stats['types'])
not_learned = train & val

for name in systems:
    data = name_to_stats_path(name)
    not_learned -= set(data['types'])

global_train_ranking = get_top_n_omitted(train_stats, not_learned, n=ranking_length)
global_val_ranking   = get_top_n_omitted(val_stats, not_learned, n=ranking_length)


################################################################################
# Local ranking

def occurrences_above_n(results, n=2):
    "Frequency filter for results."
    # Multiply n by number of systems.
    n = n * len(systems)
    return [(ratio, occurrences, word) for ratio, occurrences, word in results
                                       if occurrences > n]

def missed_ratios(total_missed, total_recalled, filtering=False, n=2):
    "Compute missed ratios for local ranking."
    # Create set of all words in importance class 5.
    all_words = set()
    all_words.update(total_missed.keys())
    all_words.update(total_recalled.keys())
    
    # Create container.
    all_results = []
    for word in all_words:
        # Compute ratio
        occurrences = total_missed[word] + total_recalled[word]
        ratio = float(total_missed[word])/occurrences
        # Create tuple in sorting order.
        # Sorting of tuples in Python means that if the first element isn't decisive, the second element is used to break the tie.
        result = (ratio, occurrences, word)
        # Add to the list.
        all_results.append(result)
    
    if filtering:
        all_results = occurrences_above_n(all_results, n)
    
    # Sort the list
    sorted_results = sorted(all_results, reverse=True)
    return sorted_results

def list_from_ratios(ratios, n):
    print('Ratios', [ratio for ratio, occurrences, word in ratios][:n])
    return [word for ratio, occurrences, word in ratios][:n]

local_recall = load_json('./Data/Output/local_recall.json')

total_missed = Counter()
total_recalled = Counter()

for system in systems:
    recalled_counter, missed_counter = local_recall[system]['counts']
    total_missed.update(missed_counter['5'])
    total_recalled.update(recalled_counter['5'])

ratios = missed_ratios(total_missed, total_recalled, filtering=False)
ratios_10 = missed_ratios(total_missed, total_recalled, filtering=True, n=10)

local_absolute = list_from_counts(total_missed.most_common(ranking_length))
local_relative = list_from_ratios(ratios, ranking_length)
local_relative_10 = list_from_ratios(ratios_10, ranking_length)

################################################################################
# Generating the table.

headers = ['', 'Train', 'Val', 'Absolute', 'Relative', 'Relative10']
ranks = range(1,ranking_length+1)
rows = list(zip(ranks,
                global_train_ranking,
                global_val_ranking,
                local_absolute,
                local_relative,
                local_relative_10))

table = tabulate(rows, headers, tablefmt='latex_booktabs')

# Postprocess table.
table = table.replace('\\toprule', '\\toprule \n & \multicolumn{2}{c}{Global ranking} & \multicolumn{3}{c}{Local ranking} \\\\')
table = table.replace('Relative10','Relative$_{10}$')
ranks = {' ' + str(i) + ' &': ' ' + str(i)+'. &' for i in ranks}
for source, target in ranks.items():
    table = table.replace(source,target)

print(table)

with open('./Data/Output/ranking_table.txt', 'w') as f:
    f.write(table)
    f.write('\n\n')


rows = list(zip(global_train_ranking,global_val_ranking))
table = tabulate(rows, ['Train', 'Val'], tablefmt='latex_booktabs')
table = table.replace('\\toprule', '\\toprule \n\multicolumn{2}{c}{Global ranking}\\\\')
print()
print(table)

with open('./Data/Output/ranking_table.txt', 'a') as f:
    f.write(table)
    f.write('\n\n')


rows = list(zip(local_absolute,local_relative,local_relative_10))
table = tabulate(rows, ['Absolute', 'Relative', 'Relative10'], tablefmt='latex_booktabs')
table = table.replace('\\toprule', '\\toprule \n\multicolumn{3}{c}{Local ranking}\\\\')
table = table.replace('Relative10', 'Relative$_{10}$')
print()
print(table)

with open('./Data/Output/ranking_table.txt', 'a') as f:
    f.write(table)
    f.write('\n\n')
