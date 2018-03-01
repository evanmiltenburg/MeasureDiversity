from methods import index_from_file, mapping_from_file, save_json
from collections import Counter, defaultdict

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_style("white")
sns.set_context('paper', font_scale=7)
my_palette = sns.color_palette("cubehelix", 10)
sns.set_palette(my_palette)

################################################################################
# Helper function.

def content_pos(pos):
    "Determine whether a POS-tag belongs to a content word."
    return any([pos.startswith('JJ'), # Adjective
                pos.startswith('NN'), # Noun
                pos.startswith('VB'), # Verb
                pos.startswith('RB')  # Adverb
                ])


def name_to_mapping(name):
    "Get mapping based on system name."
    base = './Data/Systems/'
    path = base + name + '/Val/annotated.json'
    return mapping_from_file(path)

################################################################################
# Local recall score.

def local_recall_scores(generated, ref_data):
    """
    Produce local recall score, given the generated descriptions and the references.
    """
    recalled = defaultdict(int)
    total = defaultdict(int)
    for image, references in ref_data.items():
        # Count the content words in each of the reference descriptions.
        word_counter = Counter()
        for reference in references:
            words = {word for word, tag in reference if content_pos(tag)}
            word_counter.update(words)
        # Create a set of generated words, based on the generated description.
        generated_words = set(generated[image])
        # Loop over the words, and their counts in the reference data.
        for word, count in word_counter.items():
            # We categorize these words by their frequency in the references.
            # In other words: how many descriptions contain this word?
            # If a word is produced by more annotators, we consider it to be more important.
            total[count] += 1
            # Check whether the word was actually generated and, if so, update the count.
            if word in generated_words:
                recalled[count] += 1
    # The result is a fraction for each of the frequency classes, indicating the local retrieval score.
    return [float(recalled[count])/total[count] for count in [1,2,3,4,5]]

################################################################################
# Local ranking.

def local_recall_counts(generated, ref_data):
    """
    Get local recall counts, with a counter for both recalled and missed words,
    split by frequency class.
    """
    recalled_counter = defaultdict(Counter)
    missed_counter = defaultdict(Counter)
    for image, references in ref_data.items():
        # Count the content words in each of the reference descriptions.
        word_counter = Counter()
        for reference in references:
            words = {word for word, tag in reference if content_pos(tag)}
            word_counter.update(words)
        # Create a set of generated words, based on the generated description.
        generated_words = set(generated[image])
        # Loop over the words, and their counts in the reference data.
        for word, count in word_counter.items():
            # We categorize these words by their frequency in the references.
            # In other words: how many descriptions contain this word?
            # If a word is produced by more annotators, we consider it to be more important.
            if word in generated_words:
                # If the word is actually produced, we count that specific word and its frequency class.
                recalled_counter[count][word] += 1
            else:
                # If it's not actually produced, we also count it using a separate counter.
                missed_counter[count][word] += 1
    return recalled_counter, missed_counter

################################################################################
# Plot the scores.

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


def plot_scores(results):
    fig, ax = plt.subplots(figsize=(32,20))
    lw = 8.0
    ms = 25.0
    
    score_index = {name: entry['scores'] for name, entry in results.items()}
    ordered_systems = sorted(score_index.items(), key=lambda pair:pair[1][4], reverse=True)
    
    for name, scores in ordered_systems:
        # nums = list(reversed(range(1,11)))
        # plt.plot(entry['percentiles']['val_scores'],nums,'o-',label=name,linewidth=5.0,markersize=15.0)
        nums = range(1,6)
        # Turn fractions into percentages.
        scores = [score*100 for score in scores]
        # Plot
        plt.plot(nums, scores,'o-',label=system2label[name],linewidth=lw,markersize=ms, color=system2color[name])

    labels = [system2label[name] for name,_ in ordered_systems]
    legend_markers = [Line2D(range(1), range(1),
                         linewidth=0,   # Invisible line
                         marker='o',
                         markersize=40,
                         markerfacecolor=system2color[name]) for name,_ in ordered_systems]
    plt.legend(legend_markers, labels, numpoints=1, loc=2, handletextpad=-0.3, bbox_to_anchor=(0, 1.1))
    
    # labels = ['-'.join(map(str,tup)) for tup in zip(range(0,100,10),range(10,110,10))]
    # labels = list(reversed(labels))
    # labels = [str(i) for i in range(1,11)]
    plt.xticks(range(1,6))
    plt.yticks(range(10,90,10))
    plt.tick_params(direction='in', length=10, width=4, bottom=True, left=True)
    plt.ylabel('Percent')
    plt.xlabel('Importance class')
    sns.despine()
    plt.savefig('./Data/Output/local_recall.pdf')


################################################################################
# Compute all the stats.

val_index = index_from_file('./Data/COCO/Processed/tagged_val2014.json', tagged=True, lower=True)

systems = ['Dai-et-al-2017',
           'Liu-et-al-2017',
           'Mun-et-al-2017',
           'Shetty-et-al-2016',
           'Shetty-et-al-2017',
           'Tavakoli-et-al-2017',
           'Vinyals-et-al-2017',
           'Wu-et-al-2016',
           'Zhou-et-al-2017']

all_results = dict()
for system in systems:
    print('Processing:', system)
    generated = name_to_mapping(system)
    system_results = dict(scores = local_recall_scores(generated, val_index),
                          counts = local_recall_counts(generated, val_index))
    all_results[system] = system_results

plot_scores(all_results)
save_json(all_results, './Data/Output/local_recall.json')
