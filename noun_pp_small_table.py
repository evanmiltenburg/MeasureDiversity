import json
from tabulate import tabulate

with open('Data/Output/nouns_pps.json') as f:
    data = json.load(f)


system2label  = {'Dai-et-al-2017': 'Dai et al. 2017',
                 'Liu-et-al-2017': 'Liu et al. 2017',
                 'Mun-et-al-2017': 'Mun et al. 2017',
                 'Shetty-et-al-2016': 'Shetty et al. 2016',
                 'Shetty-et-al-2017': 'Shetty et al. 2017',
                 'Tavakoli-et-al-2017': 'Tavakoli et al. 2017',
                 'Vinyals-et-al-2017': 'Vinyals et al. 2017',
                 'Wu-et-al-2016': 'Wu et al. 2016',
                 'Zhou-et-al-2017': 'Zhou et al. 2017'}

def get_system_row(system):
    return [system2label[system],
            '{:.2f}'.format(data[system]['compound_data']['compound_ratio']),
            len(data[system]['compound_data']['counts_by_length']['2']),
            '{:.2f}'.format(data[system]['pp_data']['prep_ratio']),
            len(data[system]['pp_data']['pp_counts_by_length']['1'])]

def get_val_row(data):
    scores = dict(compound_ratio=[], compound_counts=[], pp_ratio=[], pp_counts=[])
    for compound_data in data['val']['compound_data']:
        scores['compound_ratio'].append(compound_data['compound_ratio'])
        scores['compound_counts'].append(len(compound_data['counts_by_length']['2']))
    for pp_data in data['val']['pp_data']:
        scores['pp_ratio'].append(pp_data['prep_ratio'])
        scores['pp_counts'].append(len(pp_data['pp_counts_by_length']['1']))
    for key, val in scores.items():
        scores[key] = sum(val)/len(val)
    return ['Validation data',
            '{:.2f}'.format(scores['compound_ratio']),
            round(scores['compound_counts']),
            '{:.2f}'.format(scores['pp_ratio']),
            round(scores['pp_counts'])]


mles  = ['Liu-et-al-2017','Mun-et-al-2017','Shetty-et-al-2016','Tavakoli-et-al-2017',
         'Vinyals-et-al-2017','Wu-et-al-2016','Zhou-et-al-2017']
gans = ['Dai-et-al-2017', 'Shetty-et-al-2017']

mle_rows = [get_system_row(system) for system in mles]
gan_rows = [get_system_row(system) for system in gans]
val_row  = [get_val_row(data)]

all_rows = mle_rows + gan_rows + val_row

table = tabulate(all_rows, headers=['Ratio', 'Types-2', 'Ratio', 'Types-1'], tablefmt='latex_booktabs')
table = table.replace('\\toprule', '\\toprule \n & \multicolumn{2}{c}{Compound stats} & \multicolumn{2}{c}{PP stats}\\\\ \n \cmidrule(r){2-3} \cmidrule(l){4-5} \n')
table = table.replace('&    0.3  &','&    0.30 &')
table = table.replace(' Dai et al. 2017', '\\midrule\n Dai et al. 2017')
table = table.replace(' Validation data', '\\midrule\n Validation data')
print(table)

with open('Data/Output/noun_pp_small_table.txt','w') as f:
    f.write(table)
