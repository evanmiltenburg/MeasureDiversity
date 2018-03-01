from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context('paper', font_scale=7)
sns.set_palette(sns.color_palette("cubehelix", 4))

from methods import load_json, cut_curve, curve_to_coords, average_curves

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


def plot(val_curve, system_curves, best_worst, val_label='Val', filename='./Data/Output/ttr_curve_reduced.pdf'):
    fig, ax = plt.subplots(figsize=(34,20))
    lw = 8.0
    x,y = curve_to_coords(val_curve)
    ax = plt.plot(x,y,label=val_label, linewidth=lw)
    system_curves = sorted(system_curves.items(),
                           key=lambda item:item[1][max(item[1])], # Sort by highest value at the max X-value.
                           reverse=True)                          # In decreasing order.
    # plot best
    bx, by = curve_to_coords(best_worst['best'])
    #plt.plot(bx, by, color='gainsboro')
    # plot worst
    wx, wy = curve_to_coords(best_worst['worst'])
    #plt.plot(wx, wy, color='gainsboro')
    plt.fill_between(x, by, wy, color='gainsboro', alpha='0.5')
    
    for name, curve in system_curves:
        x,y = curve_to_coords(curve)
        plt.plot(x,y,label=name, linewidth=lw)
    plt.ylabel('Types')
    plt.xlabel('Tokens')
    sns.despine()
    plt.tick_params(direction='in', length=10, width=4, bottom=True, left=True)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

MLE_systems = {'Liu et al. 2017': load_curve('Liu-et-al-2017'),
               'Mun et al. 2017': load_curve('Mun-et-al-2017'),
               'Shetty et al. 2016': load_curve('Shetty-et-al-2016'),
               'Tavakoli et al. 2017': load_curve('Tavakoli-et-al-2017'),
               'Vinyals et al. 2017': load_curve('Vinyals-et-al-2017'),
               'Wu et al. 2016': load_curve('Wu-et-al-2016'),
               'Zhou et al. 2017': load_curve('Zhou-et-al-2017')}

to_plot = dict()
to_plot['Dai et al. 2017']          = load_curve('Dai-et-al-2017')
to_plot['Shetty et al. 2017']       = load_curve('Shetty-et-al-2017')
to_plot['Average of other systems'] = average_curves(MLE_systems.values())

best_worst = {'best': MLE_systems['Zhou et al. 2017'],
              'worst': MLE_systems['Liu et al. 2017']}

val_stats = load_json('./Data/COCO/Processed/val_stats.json')
val_curve = get_curve(val_stats)

plot(val_curve, to_plot, best_worst, val_label='Validation data')
