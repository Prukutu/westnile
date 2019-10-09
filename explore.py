import pickle
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('~/Documents/CCNY/Research/mpl_styles/presentations')

df = pickle.load(open('data.p', 'rb'))

# Group by species and see how many instances of WNV for each
# species_incidence = df.groupby(by='Species')['WnvPresent'].sum().sort_values()
# color = '#009688'
# fsize = (16, 4)
#
# fig1, ax1 = plt.subplots(figsize=fsize)
# sns.barplot(x=species_incidence.index,
#             y=species_incidence.values,
#             color=color,
#             ax=ax1)
#
# ax1.set_xticklabels(species_incidence.index,
#                     rotation=45,
#                     ha='right')
# ax1.set_title('WNV Presence by Species',
#               loc='left',
#               fontsize=30)
# fig1.savefig('incidence_by_species.png', bbox_inches='tight')

# Only Pipiens, Restuans, and the Pipiens/Restuans category have recorded
# presence of the virus. Let's reduce the dataset to only those three species

# Weather conditions should affect the mosquito population.
# Ciota et al. (2014) found that warmer temps leads to increased populations,
# although the effect diminishes past 24 degC. Let's see if that holds for
# our data.

# Let's test which of the temperature measures is most useful, if any.
# We're naively assuming that temperature on the day of measurement is an
# important factor in mosquito populations. It could be likely that either
# a lagged temperature or a moving average is more useful.
#
# my_species = ('CULEX PIPIENS', 'CULEX RESTUANS', 'CULEX PIPIENS/RESTUANS')
#               # 'CULEX PIPIENS/RESTUANS')
# my_species = ('CULEX PIPIENS',)
# tempvars = ('Tmin', 'Tavg', 'Tmax')
# colors = ('#f44336', '#3F51B5', '#2E7D32')
# alpha = 0.5
# fig2, ax2 = plt.subplots(ncols=len(tempvars), figsize=fsize)
# #
# # for vname, ax in zip(tempvars, ax2):
# #     for n, species in enumerate(my_species):
# #         x = df.where(df['Species'] == species)[vname].dropna()
# #         y = df.where(df['Species'] == species)['NumMosquitos'].dropna()
# #
# #         ax.scatter(x, y/50.0,
# #                    color=colors[n],
# #                    alpha=0.5,
# #                    edgecolors='none')
#
# fig2.savefig('temperature-population.png', bbox_inches='tight')

fig3, ax3 = plt.subplots(figsize=(16, 9))

corrs = df.drop(columns=['Station',
                         'Depart', 'Heat', 'Cool',
                         'DewDiff', 'ResultDir', 'year']).corr()
im3 = sns.heatmap(corrs.round(2), cmap='RdBu', annot=True, cbar=False)

fig3.savefig('heatmap.png', bbox_inches='tight')


pop_bymonth = df.groupby(by='month').mean()['NumMosquitos']
pop_byyear = df.groupby(by='year').mean()['NumMosquitos']

fig, ax = plt.subplots()
sns.barplot(x=pop_bymonth.index, y=pop_bymonth.values, color='#009688')
ax.set_title('Avg Mosquitos per trap record',
             fontsize=16,
             loc='left')
fig.savefig('popmontly.png', bbox_inches='tight')

fig, ax = plt.subplots()
sns.barplot(x=pop_byyear.index, y=pop_byyear.values, color='#009688')
ax.set_title('Avg Mosquitos per trap record',
             fontsize=16,
             loc='left')
fig.savefig('popyearly.png', bbox_inches='tight')
