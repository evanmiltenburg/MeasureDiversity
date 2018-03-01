import json

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

sns.set_style("white")
sns.set_context('paper', font_scale=7)
my_palette = sns.cubehelix_palette(10, start=.8, rot=-.95)#sns.color_palette("cubehelix", 10)
sns.set_palette(my_palette)

system2label  = {'Dai-et-al-2017': 'Dai et al. 2017',
                 'Liu-et-al-2017': 'Liu et al. 2017',
                 'Mun-et-al-2017': 'Mun et al. 2017',
                 'Shetty-et-al-2016': 'Shetty et al. 2016',
                 'Shetty-et-al-2017': 'Shetty et al. 2017',
                 'Tavakoli-et-al-2017': 'Tavakoli et al. 2017',
                 'Vinyals-et-al-2017': 'Vinyals et al. 2017',
                 'Wu-et-al-2016': 'Wu et al. 2016',
                 'Zhou-et-al-2017': 'Zhou et al. 2017'}


with open('Data/Output/nouns_pps.json') as f:
    data = json.load(f)

def get_val(data):
    val = {2:[], 3:[], 4:[]}
    for compound_data in data['val']['compound_data']:
        val[2].append(compound_data['compound_lengths']['2'])
        val[3].append(compound_data['compound_lengths']['3'])
        val[4].append(compound_data['compound_lengths']['4'])
    for i, results in val.items():
        val[i] = round(sum(results)/len(results))
    return val

# ['compound_lengths']['2']

to_plot = dict(system=[],
               length=[],
               number=[])

for system, label in system2label.items():
    to_plot['system'].extend([label] * 3)
    to_plot['length'].extend([2,3,4])
    to_plot['number'].extend([data[system]['compound_data']['compound_lengths'].get(i,0)
                              for i in ['2','3','4']])

val = get_val(data)
to_plot['system'].extend(['zzzval'] * 3)
to_plot['length'].extend([2,3,4])
to_plot['number'].extend([val[i] for i in [2,3,4]])

df = pd.DataFrame(to_plot)

fig, ax = plt.subplots(figsize=(28,20))
ax = sns.barplot(x='length', y='number', hue='system', data=df)
ax.set_yscale('log')

labels = list(system2label.values()) + ['Validation data']
legend_markers = [Line2D(range(1), range(1),
                     linewidth=0,   # Invisible line
                     marker='o',
                     markersize=40,
                     markerfacecolor=my_palette[i]) for i, name in enumerate(labels)]
plt.legend(legend_markers, labels, numpoints=1, loc=1, ncol=2, handletextpad=-0.3, bbox_to_anchor=(1.0245, 1.07), columnspacing=0)

sns.despine()
plt.tick_params(direction='in', length=10, width=4, bottom=True, left=True)
plt.tight_layout()
plt.xlabel('Compound length')
plt.ylabel('Number of tokens')
plt.savefig('Data/Output/compound_lengths.pdf')
