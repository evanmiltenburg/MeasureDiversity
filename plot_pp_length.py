import json

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

sns.set_style("white")
sns.set_context('paper', font_scale=7)
my_palette = sns.cubehelix_palette(10, start=.8, rot=-.95)#sns.color_palette("cubehelix", 10)

sns.set_palette(my_palette)

# No need for a legend if there's another figure using the same color scheme.
display_legend = False

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
    val = {1:[], 2:[], 3:[], 4:[], 5:[]}
    for pp_data in data['val']['pp_data']:
        val[1].append(pp_data['level_counter']['1'])
        val[2].append(pp_data['level_counter']['2'])
        val[3].append(pp_data['level_counter']['3'])
        val[4].append(pp_data['level_counter']['4'])
        val[5].append(pp_data['level_counter']['5'])
    for i, results in val.items():
        val[i] = round(sum(results)/len(results))
    return val

to_plot = dict(system=[],
               length=[],
               number=[])

for system, label in system2label.items():
    to_plot['system'].extend([label] * 5)
    to_plot['length'].extend([1,2,3,4,5])
    to_plot['number'].extend([data[system]['pp_data']['level_counter'].get(i,0)
                              for i in ['1','2','3','4','5']])

val = get_val(data)
to_plot['system'].extend(['zzzval'] * 5)
to_plot['length'].extend([1,2,3,4,5])
to_plot['number'].extend([val[i] for i in [1,2,3,4,5]])

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
legend = plt.legend(legend_markers, labels, numpoints=1, loc=1, ncol=2, handletextpad=-0.3, bbox_to_anchor=(1.05, 1.09), columnspacing=0)

if not display_legend:
    legend.remove()

sns.despine()
plt.tick_params(direction='in', length=10, width=4, bottom=False, left=True)
plt.tight_layout()
plt.xlabel('PP depth')
plt.ylabel('Number of tokens')
plt.savefig('Data/Output/pp_depths.pdf')
