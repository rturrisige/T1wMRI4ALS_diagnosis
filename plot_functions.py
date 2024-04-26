from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

orig_cmap = plt.cm.Blues
colors = orig_cmap(np.linspace(0.25, 1))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('mycmap', colors)


def cl_mean(d, label, saver_path, name, outcome='save'):
    mp1 = round(np.mean(d['Precision Positive Class']), 2)
    p1 = f'{mp1}'
    mp0 = round(np.mean(d['Precision Negative Class']), 2)
    p0 = f'{mp0}'
    mr1 = round(np.mean(d['Recall Positive Class']), 2)
    r1 = f'{mr1}'
    mr0 = round(np.mean(d['Recall Negative Class']), 2)
    r0 = f'{mr0}'
    mf1 = round(np.mean(d['F1-score Positive Class']), 2)
    f1 = f'{mf1}'
    mf0 = round(np.mean(d['F1-score Negative Class']), 2)
    f0 = f'{mf0}'
    ma1 = round(np.mean(d['AUC Positive Class']), 2)
    a1 = f'{ma1}'
    ma0 = round(np.mean(d['AUC Negative Class']), 2)
    a0 = f'{ma0}'
    map1 = round(np.mean(d['AUPRC Positive Class']), 2)
    ap1 = f'{map1}'
    map0 = round(np.mean(d['AUPRC Negative Class']), 2)
    ap0 = f'{map0}'
    data = pd.DataFrame({'Class': label, 'Precision': [mp0, mp1],
                         'Recall': [mr0, mr1], 'F1-score': [mf0, mf1],
                         'AUC': [ma0, ma1], 'AUPRC': [map0, map1]})
    data = data.set_index('Class')
    lab = pd.DataFrame({'Precision': [p0, p1], 'Recall': [r0, r1],
                        'F1-score': [f0, f1], 'AUC': [a0, a1], 'AUPRC': [ap0, ap1]})
    plt.figure(figsize=(10,3.5))
    ax = sns.heatmap(data, annot=lab, fmt='', square=True, cmap=cmap)
    ax.set(ylabel='')
    plt.title('Classification Report')
    if outcome == 'show':
        plt.show()
    else:
        plt.savefig(f'{saver_path}/{name}_cl.png', bbox_inches='tight')
        plt.close()


def cm_sum(d, label, save_path, name, outcome='save'):
    plt.figure()
    mtp = sum(d['True Positive'])
    tp = f'{mtp}'
    mtn = sum(d['True Negative'])
    tn = f'{mtn}'
    mfp = sum(d['False Positive'])
    fp = f'{mfp}'
    mfn = sum(d['False Negative'])
    fn = f'{mfn}'
    data = pd.DataFrame({'True Labels': label, f'{label[0]}': [mtn, mfn],
                         f'{label[1]}': [mfp, mtp]})
    data = data.set_index('True Labels')
    lab = pd.DataFrame({'column1': [tn, fn], 'column2': [fp, tp]})
    ax = sns.heatmap(data, annot=lab, fmt='', square=True, cmap=cmap, cbar=False)
    sns.set(font_scale=2)
    ax.set(xlabel='Predicted Labels')
    plt.title('Confusion Matrix')
    plt.subplots_adjust(left=0.1,
                        bottom=0.2,
                        right=0.9,
                        top=0.9)
    if outcome == 'show':
        plt.show()
    else:
        plt.savefig(f'{save_path}/{name}_cm_sum.png', bbox_inches='tight')
        plt.close()

