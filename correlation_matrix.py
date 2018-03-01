import json
from methods import load_json
from tabulate import tabulate
from scipy.stats import spearmanr
from itertools import combinations
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

absolute=True

sns.set_context('paper', font_scale=7)
plt.subplots(figsize=(34,12))

def load_system_stats(name):
    "Load system stats based on the system name."
    base = './Data/Systems/'
    path = base + name + '/Val/stats.json'
    return load_json(path)

systems = ['Dai-et-al-2017', 'Liu-et-al-2017', 'Mun-et-al-2017', 'Shetty-et-al-2016',
           'Shetty-et-al-2017', 'Tavakoli-et-al-2017', 'Vinyals-et-al-2017',
           'Wu-et-al-2016', 'Zhou-et-al-2017']

# Load the data
system_stats = {sys_name: load_system_stats(sys_name) for sys_name in systems}
global_recall = load_json('./Data/Output/global_recall.json')
local_recall = load_json('./Data/Output/local_recall.json')

# Values to be correlated.
system_keys = ["average_sentence_length", 'std_sentence_length', "num_types", "type_token_ratio", 'bittr', 'percentage_novel']

# Let's first index all scores by system.
# This is easiest to inspect, and we don't care about efficiency here.
result_rows = dict()
for system in systems:
    result_rows[system] = [system_stats[system][key] for key in system_keys]
    
    # Add local and global recall scores from their separate files.
    global_recall_score = global_recall[system]['score']
    local_recall_score  = local_recall[system]['scores'][-1]
    
    result_rows[system].append(global_recall_score)
    result_rows[system].append(local_recall_score)

cats = ["ASL", "SDSL", "Types", "TTR1", 'TTR2', 'Novel', 'Cov', 'Loc5']
# Convert lists of different scores into lists of the same scores.
# The scores in those lists are ordered by system.
scores = list(zip(*result_rows.values()))
# Map score names to lists containing these scores.
to_correlate = dict(zip(cats,scores))

# Prepare dict which will be converted to DataFrame.
correlations = dict(cat1=[], cat2=[], corr=[])

# Correlation of items with themselves is always 1.
for cat in cats:
    correlations['cat1'].append(cat)
    correlations['cat2'].append(cat)
    correlations['corr'].append(1)

# Now add combinations of categories..
for cat1, cat2 in combinations(cats,2):
    list1 = to_correlate[cat1]
    list2 = to_correlate[cat2]
    corr, pvalue = spearmanr(list1, list2)
    
    # Order 1
    correlations['cat1'].append(cat1)
    correlations['cat2'].append(cat2)
    correlations['corr'].append(abs(corr) if absolute else corr)
        
    
    # Order 2 (correlation is symmetric)
    correlations['cat2'].append(cat1)
    correlations['cat1'].append(cat2)
    correlations['corr'].append(abs(corr) if absolute else corr)

# Convert to DataFrame
df = pd.DataFrame.from_dict(correlations)

# Pivot to obtain the proper table, with Cats X Cats.
df = df.pivot("cat1", "cat2", "corr")

# Plot
data = df.drop(['Cov', 'Loc5']).drop(['Cov', 'Loc5'], axis=1)
data = data[cats[:-2]]
data = data.reindex(cats[:-2])
ax = sns.heatmap(data, annot=True, cbar=False, fmt='.2f')

# Adjust plot.
ax.set_xlabel('')
ax.set_ylabel('')
plt.yticks(rotation=0)
plt.tick_params(axis='x', labeltop=True, labelbottom=False)

# Save plot
plt.savefig('./Data/Output/heatmap.pdf')

################################################################################
# Second heatmap.

plt.clf()
plt.subplots(figsize=(34,6))

# Same table, but drop all rows but cov and loc5.
df = df.drop(["ASL", "SDSL", "Types", "TTR1", 'TTR2', 'Novel'])
df = df[cats]

# Plot
ax = sns.heatmap(df, annot=True, cbar=False, fmt='.2f')

# Adjust plot.
ax.set_xlabel('')
ax.set_ylabel('')
plt.yticks(rotation=0)
plt.tick_params(axis='x', labeltop=True, labelbottom=False)
plt.tight_layout()

# Save plot
plt.savefig('./Data/Output/heatmap_cov_loc5.pdf')
