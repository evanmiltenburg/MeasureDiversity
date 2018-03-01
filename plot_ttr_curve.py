from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context('paper', font_scale=7)
sns.set_palette(sns.color_palette("cubehelix", 10))

from methods import load_json, cut_curve, curve_to_coords

def get_curve(stats, n=50000):
    "Prepare curve for plotting"
    curve = {int(key): val for key,val in stats['ttr_curve'].items()}
    cut_curve(curve, n)
    return curve


def load_system_stats(name):
    "Load system stats based on the system name."
    base = './Data/Systems/'
    path = base + name + '/Val/stats.json'
    return load_json(path)


def load_curve(name):
    "Wrapper to load a curve."
    stats = load_system_stats(name)
    curve = get_curve(stats)
    return curve


def plot(val_curve, system_curves, val_label='Val', legend=True, filename='./Data/Output/ttr_curve.pdf'):
    fig, ax = plt.subplots(figsize=(28,15))
    lw = 8.0
    x,y = curve_to_coords(val_curve)
    ax = plt.plot(x,y,label=val_label, linewidth=lw)
    system_curves = sorted(system_curves.items(),
                           key=lambda item:item[1][max(item[1])], # Sort by highest value at the max X-value.
                           reverse=True)                          # In decreasing order.
    for name, curve in system_curves:
        x,y = curve_to_coords(curve)
        plt.plot(x,y,label=name, linewidth=lw)
    plt.ylabel('Types')
    plt.xlabel('Tokens')
    sns.despine()
    if legend:
        lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        plt.savefig(filename)
    plt.clf()

systems = ['Dai-et-al-2017',
           'Liu-et-al-2017',
           'Mun-et-al-2017',
           'Shetty-et-al-2016',
           'Shetty-et-al-2017',
           'Tavakoli-et-al-2017',
           'Vinyals-et-al-2017',
           'Wu-et-al-2016',
           'Zhou-et-al-2017']

system_curves = {name: load_curve(name) for name in systems}

val_stats = load_json('./Data/COCO/Processed/val_stats.json')
val_curve = get_curve(val_stats)
plot(val_curve, system_curves, val_label='Val')
plot(val_curve, system_curves, val_label='Val', legend=False, filename='./Data/Output/ttr_curve_nolegend.pdf')
