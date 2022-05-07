import pandas as pd
from data.calcium_data import CalciumData

# %%
animal2 = 'PGT08'
dates2 = ['071621', '072721']

animal3 = 'PGT06'
dates3 = ['051321', '050721']

animal1_data = []
for date in dates:
    data_day = CalciumData(animal1, date, datadir)
    print((data_day.session, data_day.cells, data_day.numlicks))
    animal1_data.append(data_day)
cells1 = animal1_data[0].cells

animal2_data = []
for date in dates2:
    data_day = CalciumData(animal2, date, datadir)
    print((data_day.session, data_day.cells, data_day.numlicks))
    animal2_data.append(data_day)
cells2 = animal2_data[0].cells

animal3_data = []
for date in dates3:
    data_day = CalciumData(animal3, date, datadir)
    print((data_day.session, data_day.cells, data_day.numlicks))
    animal3_data.append(data_day)
cells3 = animal3_data[0].cells

# %% Fill Data Containers

# Animal 1 Day 1
as_zdict_day1 = stat_help.get_single_tastant_dicts('ArtSal', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
s_zdict_day1 = stat_help.get_single_tastant_dicts('Sucrose', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
n_zdict_day1 = stat_help.get_single_tastant_dicts('NaCl', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
ca_zdict_day1 = stat_help.get_single_tastant_dicts('Citric', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
q_zdict_day1 = stat_help.get_single_tastant_dicts('Quinine', cells1, [animal1_data[0]], do_zscore=True, baseline=1)
msg_zdict_day1 = stat_help.get_single_tastant_dicts('MSG', cells1, [animal1_data[0]], do_zscore=True, baseline=1)

t_as_zdict_day1 = stat_help.get_single_tastant_dicts('ArtSal', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_s_zdict_day1 = stat_help.get_single_tastant_dicts('Sucrose', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_n_zdict_day1 = stat_help.get_single_tastant_dicts('NaCl', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_ca_zdict_day1 = stat_help.get_single_tastant_dicts('Citric', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_q_zdict_day1 = stat_help.get_single_tastant_dicts('Quinine', cells2, [animal2_data[0]], do_zscore=True, baseline=1)
t_msg_zdict_day1 = stat_help.get_single_tastant_dicts('MSG', cells2, [animal2_data[0]], do_zscore=True, baseline=1)

tt_as_zdict_day1 = stat_help.get_single_tastant_dicts('ArtSal', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_s_zdict_day1 = stat_help.get_single_tastant_dicts('Sucrose', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_n_zdict_day1 = stat_help.get_single_tastant_dicts('NaCl', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_ca_zdict_day1 = stat_help.get_single_tastant_dicts('Citric', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_q_zdict_day1 = stat_help.get_single_tastant_dicts('Quinine', cells3, [animal3_data[0]], do_zscore=True, baseline=1)
tt_msg_zdict_day1 = stat_help.get_single_tastant_dicts('MSG', cells3, [animal3_data[0]], do_zscore=True, baseline=1)

day1 = []

s_day1 = pd.concat([pd.concat([v for k, v in s_zdict_day1.items()]),
                    pd.concat([v for k, v in t_s_zdict_day1.items()]),
                    pd.concat([v for k, v in tt_s_zdict_day1.items()])],
                   axis=0)
day1.append(s_day1)

s_day1.reset_index(drop=True, inplace=True)
s_day1['value'] = s_day1[49] - s_day1[0]
s_day1.sort_values(by='value', inplace=True, ascending=False)
s_day1.drop(columns='value', inplace=True)
s_day1.dropna(axis=1, inplace=True)

as_day1 = pd.concat([pd.concat([v for k, v in as_zdict_day1.items()]),
                     pd.concat([v for k, v in t_as_zdict_day1.items()]),
                     pd.concat([v for k, v in tt_as_zdict_day1.items()])],
                    axis=0).reset_index(drop=True).reindex(s_day1.index)

day1.append(as_day1)

n_day1 = pd.concat([pd.concat([v for k, v in n_zdict_day1.items()]),
                    pd.concat([v for k, v in t_n_zdict_day1.items()]),
                    pd.concat([v for k, v in tt_n_zdict_day1.items()])],
                   axis=0).reset_index(drop=True).reindex(s_day1.index)
day1.append(n_day1)

ca_day1 = pd.concat([pd.concat([v for k, v in ca_zdict_day1.items()]),
                     pd.concat([v for k, v in t_ca_zdict_day1.items()]),
                     pd.concat([v for k, v in tt_ca_zdict_day1.items()])],
                    axis=0).reset_index(drop=True).reindex(s_day1.index)

day1.append(ca_day1)

q_day1 = pd.concat([pd.concat([v for k, v in q_zdict_day1.items()]),
                    pd.concat([v for k, v in t_q_zdict_day1.items()]),
                    pd.concat([v for k, v in tt_q_zdict_day1.items()])],
                   axis=0).reset_index(drop=True).reindex(s_day1.index).dropna()
day1.append(q_day1)

msg_day1 = pd.concat([pd.concat([v for k, v in msg_zdict_day1.items()]),
                      pd.concat([v for k, v in t_msg_zdict_day1.items()]),
                      pd.concat([v for k, v in tt_msg_zdict_day1.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna()

as_zdict_day2 = stat_help.get_single_tastant_dicts('ArtSal', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
s_zdict_day2 = stat_help.get_single_tastant_dicts('Sucrose', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
n_zdict_day2 = stat_help.get_single_tastant_dicts('NaCl', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
ca_zdict_day2 = stat_help.get_single_tastant_dicts('Citric', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
q_zdict_day2 = stat_help.get_single_tastant_dicts('Quinine', cells1, [animal1_data[1]], do_zscore=True, baseline=1)
msg_zdict_day2 = stat_help.get_single_tastant_dicts('MSG', cells1, [animal1_data[1]], do_zscore=True, baseline=1)

t_as_zdict_day2 = stat_help.get_single_tastant_dicts('ArtSal', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_s_zdict_day2 = stat_help.get_single_tastant_dicts('Sucrose', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_n_zdict_day2 = stat_help.get_single_tastant_dicts('NaCl', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_ca_zdict_day2 = stat_help.get_single_tastant_dicts('Citric', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_q_zdict_day2 = stat_help.get_single_tastant_dicts('Quinine', cells2, [animal2_data[1]], do_zscore=True, baseline=1)
t_msg_zdict_day2 = stat_help.get_single_tastant_dicts('MSG', cells2, [animal2_data[1]], do_zscore=True, baseline=1)

tt_as_zdict_day2 = stat_help.get_single_tastant_dicts('ArtSal', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_s_zdict_day2 = stat_help.get_single_tastant_dicts('Sucrose', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_n_zdict_day2 = stat_help.get_single_tastant_dicts('NaCl', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_ca_zdict_day2 = stat_help.get_single_tastant_dicts('Citric', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_q_zdict_day2 = stat_help.get_single_tastant_dicts('Quinine', cells3, [animal3_data[1]], do_zscore=True, baseline=1)
tt_msg_zdict_day2 = stat_help.get_single_tastant_dicts('MSG', cells3, [animal3_data[1]], do_zscore=True, baseline=1)

s_day2 = pd.concat([pd.concat([v for k, v in s_zdict_day2.items()]),
                    pd.concat([v for k, v in t_s_zdict_day2.items()]),
                    pd.concat([v for k, v in tt_s_zdict_day2.items()])],
                   axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')

as_day2 = pd.concat([pd.concat([v for k, v in as_zdict_day2.items()]),
                     pd.concat([v for k, v in t_as_zdict_day2.items()]),
                     pd.concat([v for k, v in tt_as_zdict_day2.items()])],
                    axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')

n_day2 = pd.concat([pd.concat([v for k, v in n_zdict_day2.items()]),
                    pd.concat([v for k, v in t_n_zdict_day2.items()]),
                    pd.concat([v for k, v in tt_n_zdict_day2.items()])],
                   axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')

ca_day2 = pd.concat([pd.concat([v for k, v in ca_zdict_day2.items()]),
                     pd.concat([v for k, v in t_ca_zdict_day2.items()]),
                     pd.concat([v for k, v in tt_ca_zdict_day2.items()])],
                    axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
q_day2 = pd.concat([pd.concat([v for k, v in q_zdict_day2.items()]),
                    pd.concat([v for k, v in t_q_zdict_day2.items()]),
                    pd.concat([v for k, v in tt_q_zdict_day2.items()])],
                   axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')
msg_day2 = pd.concat([pd.concat([v for k, v in msg_zdict_day2.items()]),
                      pd.concat([v for k, v in t_msg_zdict_day2.items()]),
                      pd.concat([v for k, v in tt_msg_zdict_day2.items()])],
                     axis=0).reset_index(drop=True).reindex(s_day1.index).dropna(how='all')

day1 = []
day1.extend([as_day1, s_day1, n_day1, q_day1, msg_day1])
