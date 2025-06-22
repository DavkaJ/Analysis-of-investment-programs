import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr


data = pd.read_csv('data\\reg_data.csv', delimiter=';', skipinitialspace=True)

features = data.drop(columns=['year', 'врп'])
target = data['врп'] / 100
correlation_results = []


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
axes = axes.flatten()


for i, feature in enumerate(features.columns):
    sns.regplot(
        x=feature, y='врп', data=data, ax=axes[i],
        ci=None,
        line_kws={"color": "red", "linestyle": "--", "linewidth": 1.5}
    )
    corr, p_val = pearsonr(data[feature], target)
    axes[i].set_title(f'{feature} к врп\nr = {corr:.2f}', fontsize=11)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('врп')
    correlation_results.append([feature, round(corr, 4), round(p_val, 4)])
    axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}'))


axes[-1].axis('off')
table_data = pd.DataFrame(correlation_results, columns=['Признак', 'Корреляция', 'p-значение'])

table = axes[-1].table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    loc='center',
    cellLoc='center',
    colColours=["lightgrey"] * 3
)
table.scale(0.75, 2)
table.auto_set_font_size(False)
table.set_fontsize(13)

# Увеличение ширины первого столбца
cell_dict = table.get_celld()
for i in range(len(correlation_results) + 1):  
    cell_dict[(i, 0)].set_width(0.5)

# Подсветка строк: зелёный если p < 0.05, иначе красный
for i, row in enumerate(correlation_results):
    p_val = row[2]
    color = '#ccffcc' if p_val < 0.05 else '#ffcccc'  
    for j in range(3):
        table[(i + 1, j)].set_facecolor(color)

plt.tight_layout()
plt.show()
