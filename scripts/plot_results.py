"""results/*.csv -> results/compare.png"""
import matplotlib.pyplot as plt
import pandas as pd

acc = pd.read_csv('results/accuracy.csv')
spd = pd.read_csv('results/speed.csv')

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
acc.plot.bar(x='model', y=['mAP50', 'mAP50-95'], ax=ax[0], rot=0, title='Accuracy (all classes)', ylim=(0.7, 1.0))
acc.plot.bar(x='model', y=['abnormal_AP50', 'abnormal_AP50-95'], ax=ax[1], rot=0, title='Accuracy (abnormal)', ylim=(0.7, 1.0))
spd.pivot(index='model', columns='device', values='mean_ms').plot.bar(ax=ax[2], rot=0, title='Inference time (ms / image)')
for a in ax:
    for c in a.containers:
        a.bar_label(c, fmt='%.3g', fontsize=8)
plt.tight_layout()
plt.savefig('results/compare.png', dpi=160)
print('saved results/compare.png')
