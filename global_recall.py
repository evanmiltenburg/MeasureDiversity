from methods import load_json, save_json, chunks
from collections import Counter
from math import ceil

# External libs:
from tabulate import tabulate
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_style("white")
sns.set_context('paper', font_scale=7)
my_palette = sns.color_palette("cubehelix", 10)
sns.set_palette(my_palette)

################################################################################
# Main score

def coverage(name, target):
    """
    Compute coverage for a specific system.
    
    This function is agnostic to whether you want coverage over entire Val or only
    the set of learnable types.
    """
    base = './Data/Systems/'
    path = base + name + '/Val/stats.json'
    system = load_json(path)
    gen = set(system['types'])
    recalled = gen & target
    return {"recalled": recalled,
            "score": len(recalled)/len(target),
            "not_in_val": gen - target}

################################################################################
# Ranking

def most_frequent_omissions(recalled, ref_stats, n=None):
    """
    Rank the most frequent omissions.
    
    This function is agnostic to whether you want to use test or val as reference.
    """
    counts = Counter((word, ref_stats['total_counts'][word]) for word in recalled)
    if n:
        return counts.most_common(n)
    else:
        return counts.most_common()

################################################################################
# Percentile coverage

def chunk_retrieval_score(chunk, retrieved):
    "Compute retrieval scores for one chunk."
    overlap = set(chunk) & retrieved
    percentage = (len(overlap)/len(chunk)) * 100
    return percentage


def retrieval_scores(original, retrieved, chunk_size):
    "Compute retrieval scores for all chunks."
    return [chunk_retrieval_score(chunk, retrieved)
            for chunk in chunks(original, chunk_size)]


def percentiles(val_count_list, retrieved):
    "Compute retrieval scores for each percentile."
    val_ordered = [word for word, count in val_count_list]
    chunk_size = ceil(float(len(val_ordered))/10)
    return {'val_scores': retrieval_scores(val_ordered, retrieved, chunk_size),
            'num_percentiles': 10}


def get_count_list(stats):
    "Get count list from a ref stats file."
    c = Counter(stats['total_counts'])
    return c.most_common()

system2label  = {'Dai-et-al-2017': 'Dai et al. 2017',
                 'Liu-et-al-2017': 'Liu et al. 2017',
                 'Mun-et-al-2017': 'Mun et al. 2017',
                 'Shetty-et-al-2016': 'Shetty et al. 2016',
                 'Shetty-et-al-2017': 'Shetty et al. 2017',
                 'Tavakoli-et-al-2017': 'Tavakoli et al. 2017',
                 'Vinyals-et-al-2017': 'Vinyals et al. 2017',
                 'Wu-et-al-2016': 'Wu et al. 2016',
                 'Zhou-et-al-2017': 'Zhou et al. 2017'}

system2color = dict(zip(sorted(system2label),my_palette))

def plot_percentiles(results):
    fig, ax = plt.subplots(figsize=(28,20))
    lw = 8.0
    ms = 25.0
    ordered_systems = sorted(results.items(),
                             key=lambda pair:pair[1]['percentiles']['val_scores'][1],
                             reverse=True)
    
    plt.axvline(x=2.5, linestyle='dashed', linewidth=5, color='gray')
    plt.axvline(x=6.5, linestyle='dashed', linewidth=5, color='gray')
    
    plt.text(1.35, 90, 'A', color='gray')
    plt.text(4.3, 90, 'B', color='gray')
    plt.text(8.5, 90, 'C', color='gray')
    
    for name, entry in ordered_systems:
        # nums = list(reversed(range(1,11)))
        # plt.plot(entry['percentiles']['val_scores'],nums,'o-',label=name,linewidth=5.0,markersize=15.0)
        nums = range(1,11)
        plt.plot(nums, entry['percentiles']['val_scores'],'o-', label=system2label[name], linewidth=lw, markersize=ms, color=system2color[name])

    #plt.legend(ncol=2, loc=1, bbox_to_anchor=(1.05, 1))
    
    labels = [system2label[name] for name,_ in ordered_systems]
    legend_markers = [Line2D(range(1), range(1),
                         linewidth=0,   # Invisible line
                         marker='o',
                         markersize=40,
                         markerfacecolor=system2color[name]) for name,_ in ordered_systems]
    plt.legend(legend_markers, labels, numpoints=1, loc=1, handletextpad=-0.3, bbox_to_anchor=(1.05, 0.85))
    
    # labels = ['-'.join(map(str,tup)) for tup in zip(range(0,100,10),range(10,110,10))]
    # labels = list(reversed(labels))
    labels = [str(i * 10) for i in range(1,11)]
    plt.xticks(range(1,11), labels)
    sns.despine()
    plt.tick_params(direction='in', length=10, width=4, bottom=True, left=True)
    plt.ylabel('Coverage')
    plt.xlabel('Top N percentile')
    plt.savefig('./Data/Output/percentiles.pdf')

################################################################################
# Main definitions.

train_stats = load_json('./Data/COCO/Processed/train_stats.json')
val_stats = load_json('./Data/COCO/Processed/val_stats.json')

train     = set(train_stats['types'])
val       = set(val_stats['types'])
learnable = train & val

limit = len(learnable)/len(val)
size_limit = len(val) - len(learnable)
print(f'The limit is: {limit}. This means {size_limit} words in Val cannot be learned.')

################################################################################
# Run the script.

systems = ['Dai-et-al-2017',
           'Liu-et-al-2017',
           'Mun-et-al-2017',
           'Shetty-et-al-2016',
           'Shetty-et-al-2017',
           'Tavakoli-et-al-2017',
           'Vinyals-et-al-2017',
           'Wu-et-al-2016',
           'Zhou-et-al-2017']

# Get coverage results
coverage_results = {system:coverage(system, learnable) for system in systems}

# Add global omission ranking
for entry in coverage_results.values():
    entry['omissions'] = most_frequent_omissions(entry['recalled'],
                                                 val_stats,         # Use validation set as reference.
                                                 n=None)            # Rank everything

# Add percentile scores.
val_count_list = get_count_list(val_stats)
for entry in coverage_results.values():
    recalled = entry['recalled']
    entry['percentiles'] = percentiles(val_count_list, recalled)

plot_percentiles(coverage_results)

# Save the data
save_json(coverage_results, './Data/Output/global_recall.json')

# Show a table with the results.
table = tabulate(tabular_data=[(system, entry['score']) for system, entry in coverage_results.items()],
                 headers=['System', 'Coverage'],
                 tablefmt='latex_booktabs',
                 floatfmt='.2f')

print(table)
with open('./Data/Output/global_recall_table.txt','w') as f:
    f.write(table)
    f.write('\n\n')
    f.write(f'The limit is:  {limit}. This means {size_limit} words in Val cannot be learned.')
