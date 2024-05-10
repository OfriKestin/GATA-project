import pickle
import dill
import sys
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
import seaborn as sns
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap


# sys.path.append(r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW")
# sys.path.append(r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\files")


def calc_all_DSA_diff(path_WP, path_GP):
    # input type -> path to DSA table file result from interFLOW
    pkl_file = open(path_WP, 'rb')
    dict1 = pickle.load(pkl_file)
    pkl_file.close()
    pkl_file = open(path_GP, 'rb')
    dict2 = pickle.load(pkl_file)
    pkl_file.close()
    max_DSA_diff = 0
    scores = {}
    for i in dict1.keys():
        for j in dict1[i]:
            calc = merge_DSA(dict1[i][j], dict2[i][j])
            # print(calc, "from: " + j, "to: " + i)
            scores["from: " + j, "to: " + i] = calc
            max_DSA_diff = max(max_DSA_diff, calc)
    return max_DSA_diff


def merge_DSA(df_wp, df_gp):
    # Merge DataFrames on 'Receptor'
    merged_df = pd.merge(df_wp, df_gp, on=['Recp'], how='outer', suffixes=('_WP', '_GP'))
    # Fill missing values with '0'
    merged_df = merged_df.fillna(0)
    wp_DSA = merged_df['DSA_WP']
    gp_DSA = merged_df['DSA_GP']
    merged_df['dist'] = gp_DSA - wp_DSA
    return merged_df


def merge_lists(exp_wp, exp_gp, DSA_wp, DSA_gp, sig_wp, sig_gp):
    # Merge DataFrames on 'Ligand' and 'Receptor'
    merged_exp_df = pd.merge(exp_wp, exp_gp, on=['Ligand', 'Receptor'], suffixes=('_WP', '_GP'), how='outer')
    merged_exp_df = merged_exp_df.fillna(np.nan)
    # Iterate through columns and fill NaN with minimum value if greater than zero
    for col in merged_exp_df.columns:
        if col.endswith('_WP') or col.endswith('_GP'):
            min_val = merged_exp_df[col].min()
            if min_val > 0:
                merged_exp_df[col] = merged_exp_df[col].fillna(min_val)
    merged_exp_df['delta_ligandEXP'] = merged_exp_df['LigandExp_GP'] - merged_exp_df['LigandExp_WP']
    merge_DSA_df = merge_DSA(DSA_wp, DSA_gp)
    # Create a dictionary from merge_DSA_df with Receptor as key and dist as value
    receptor_dist_dict = dict(zip(merge_DSA_df['Recp'], merge_DSA_df['dist']))
    # Add a new 'dist' column to merged_exp_df based on the values in the dictionary
    merged_exp_df['DSA_delta'] = merged_exp_df['Receptor'].map(receptor_dist_dict)
    # Fill missing values in 'dist' column with 0
    merged_exp_df['DSA_delta'] = merged_exp_df['DSA_delta'].fillna(0)
    merged_exp_df['flag'] = 0
    rows_to_drop = []
    for index, row in merged_exp_df.iterrows():
        if row['delta_ligandEXP'] < 0:
            merged_exp_df.loc[index, 'flag'] = 1
        if (row['DSA_delta'] < 0 and row['delta_ligandEXP'] > 0) or (
                row['DSA_delta'] > 0 and row['delta_ligandEXP'] < 0):  # Opposite ligand receptor signs
            merged_exp_df = merged_exp_df.drop(index)
        if row['Receptor'] not in sig_gp and row['Receptor'] not in sig_wp:
            rows_to_drop.append(index)
    merged_exp_df = merged_exp_df.drop(rows_to_drop, errors='ignore')
    merged_exp_df['DSA_delta'] = merged_exp_df['DSA_delta'].abs()
    merged_exp_df['delta_ligandEXP'] = merged_exp_df['delta_ligandEXP'].abs()
    merged_exp_df['DSA_delta_unnormalized'] = merged_exp_df['DSA_delta'].copy()
    merged_exp_df['delta_ligandEXP_unnormalized'] = merged_exp_df['delta_ligandEXP'].copy()
    # merged_exp_df.to_csv('test.csv', index=False)
    return merged_exp_df


def merge_all_func(path_WP_list, path_GP_list, path_WP_table, path_GP_table, path_sig_wp, path_sig_gp):
    # Define the paths for pickle files
    paths = [path_WP_list, path_GP_list, path_WP_table, path_GP_table, path_sig_wp, path_sig_gp]
    # Initialize an empty list to store dictionaries
    dicts = []
    # Loop through each path and load the corresponding pickle file
    for path in paths:
        with open(path, 'rb') as pkl_file:
            dicts.append(pickle.load(pkl_file))
    # Unpack the loaded dictionaries
    dict1, dict2, dict3, dict4, dict_sig_wp, dict_sig_gp = dicts
    merge_all = pd.DataFrame()
    for i in dict1.keys():
        for j in dict1[i]:
            # print("from",j, "to", i)
            cur = merge_lists(dict1[i][j], dict2[i][j], dict3[i][j], dict4[i][j], dict_sig_wp[i][j], dict_sig_gp[i][j])
            cur['interaction'] = " ".join(["from", j, "to", i])
            merge_all = pd.concat([merge_all, cur])
    scaler = MinMaxScaler()
    columns_to_scale = ['DSA_delta', 'delta_ligandEXP']
    # Fit and transform the selected columns
    merge_all[columns_to_scale] = scaler.fit_transform(merge_all[columns_to_scale])
    merge_all['delta'] = merge_all['DSA_delta'] + merge_all['delta_ligandEXP']
    merge_all.to_csv('merge_all_T_1.csv', index=False)
    return


def permutations_func(merged_df, merge_all, amount):
    df = pd.read_csv(merge_all)
    means = []
    real_mean = merged_df['delta'].mean()
    print(real_mean)
    for i in range(1000):
        copy = merged_df.copy()
        # Sample amount% of rows from df1
        sampled_rows = copy.sample(frac=amount, random_state=i)
        # Drop the sampled rows from df1
        df1 = copy.drop(sampled_rows.index)
        # Sample amount% of rows from df2
        replacement_rows = df.sample(n=len(sampled_rows), random_state=i)
        # Concatenate the replacement rows with df1
        df1 = pd.concat([df1, replacement_rows])
        cur_mean = df1['delta'].mean()
        means.append(cur_mean)
    print(means)
    sum_larger = sum([1 for i in means if i >= real_mean])
    p = sum_larger / 1000
    return p


def calc_all_p_values(path_WP_list, path_GP_list, path_WP_table, path_GP_table, merge_all, path_sig_wp, path_sig_gp):
    # input type -> paths to: WP list, GP list, WP Table, GP table, all interactions csv, significant receptors dicts
    paths = [path_WP_list, path_GP_list, path_WP_table, path_GP_table, path_sig_wp, path_sig_gp]
    # Initialize an empty list to store dictionaries
    dicts = []
    # Loop through each path and load the corresponding pickle file
    for path in paths:
        with open(path, 'rb') as pkl_file:
            dicts.append(pickle.load(pkl_file))
    # Unpack the loaded dictionaries
    dict1, dict2, dict3, dict4, dict_sig_wp, dict_sig_gp = dicts
    means = {}
    p_values = {}
    df = pd.read_csv(merge_all)
    for i in dict1.keys():
        for j in dict1[i]:
            cur = df[df['interaction'] == " ".join(["from", j, "to", i])]
            cur_len = cur.shape[0]
            p_val = permutations_func(cur, merge_all, 0.5)
            p_val = round(p_val, 3)
            p_values["from " + j, "to " + i] = p_val
            mean = round(cur['delta'].mean(), 3)
            means["from " + j, "to " + i] = mean, cur_len
    sorted_means = dict(sorted(means.items(), key=lambda item: item[1][0], reverse=True))
    sorted_p_values = dict(sorted(p_values.items(), key=lambda item: item[1]))
    with open('delta_interactions_T_1.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Interaction', 'Value', 'Interactions Num'])  # Writing header
        for interaction, (value, interactions_num) in sorted_means.items():
            csv_writer.writerow([interaction, value, interactions_num])
    with open('p_val_T_1.csv', 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Interaction', 'Value'])  # Writing header
        for interaction in sorted_p_values.items():
            csv_writer.writerow([interaction[0], interaction[1]])
    return sorted_means


def create_diverging_dotplot_1(file_path, title):
    df = pd.read_csv(file_path)
    df = df[df['interaction'] == title]
    # Ensure that the column name is consistent throughout
    df['delta'] = df.apply(lambda row: row['delta'] * -1 if row['flag'] == 1 else row['delta'], axis=1)
    df.sort_values('delta', inplace=True)
    cmap = LinearSegmentedColormap.from_list('custom_colormap', ['#3498db', '#8e44ad', '#e74c3c'])
    norm = plt.Normalize(df['delta'].min(), df['delta'].max())
    df['colors'] = [cmap(norm(x)) for x in df['delta']]
    df.reset_index(inplace=True)
    # Draw plot
    plt.figure(figsize=(14, 18), dpi=80)
    plt.scatter(df.delta, df.index, s=450, alpha=.6, color=df.colors, label='Delta Values')  # Use 'df.delta' here

    # Add legend
    legend_labels = [
        plt.Line2D([0], [0], marker='o', color='w', label='WT', markerfacecolor='#3498db', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Middle', markerfacecolor='#8e44ad', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='GATA mut', markerfacecolor='#e74c3c', markersize=10)
    ]

    plt.legend(handles=legend_labels, loc='lower right')

    for x, y, tex in zip(df.delta, df.index, df.delta):
        t = plt.text(x, y, round(tex, 1), horizontalalignment='center',
                     verticalalignment='center', fontdict={'color': 'white'})

    # Decorations
    plt.gca().spines["top"].set_alpha(.3)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.3)
    plt.gca().spines["left"].set_alpha(.3)
    plt.yticks(df.index, [f'{ligand} - {receptor}' for ligand, receptor in zip(df.Ligand, df.Receptor)])
    plt.title(title + ' deltas diverging dot plot ', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-1.05, 1.05)
    plt.savefig(title + 'diverging_dotplot.pdf')
    plt.show()


def create_diverging_dotplot(file_path, title):
    df = pd.read_csv(file_path)
    df['delta'] = df.apply(lambda row: row['delta'] * -1 if row['flag'] == 1 else row['delta'], axis=1)
    df.sort_values('delta', inplace=True)
    # Remove duplicates from the 'Ligand' column
    unique_ligands = df['Ligand'].drop_duplicates()
    df['colors'] = ['red' if x < 0 else 'darkgreen' for x in df['delta']]
    # Create a mapping from ligand to a unique integer for y-axis positioning
    ligand_mapping = {ligand: idx for idx, ligand in enumerate(unique_ligands)}

    # Map the ligands to integers and add a new column 'y_position' to the DataFrame
    df['y_position'] = df['Receptor'].map(ligand_mapping)

    df.sort_values('delta', inplace=True)
    df.reset_index(inplace=True)

    # Draw plot
    plt.figure(figsize=(14, 18), dpi=80)
    plt.scatter(df.DSA_delta, df['y_position'], s=450, alpha=.6, color=df.colors)
    for x, y, tex in zip(df.delta, df['y_position'], df.delta):
        t = plt.text(x, y, round(tex, 1), horizontalalignment='center',
                     verticalalignment='center', fontdict={'color': 'white'})

    # Decorations
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.3)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.3)
    plt.gca().spines["left"].set_alpha(.3)

    # Use unique ligands for y-axis ticks
    plt.yticks(range(len(unique_ligands)), unique_ligands, fontsize=20)

    plt.title(title + 'DSA deltas diverging dot plot ', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-1.05, 1.05)
    plt.show()


def create_RidgePlot(file, wanted_interactions):
    data = pd.read_csv(file)

    # Multiply 'delta' by -1 where 'flag' is 1 in the DataFrame
    filtered_data = data[data['interaction'].isin(wanted_interactions)]
    filtered_data['delta'] = filtered_data.apply(lambda row: row['delta'] * -1 if row['flag'] == 1 else row['delta'],
                                                 axis=1)

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), "axes.labelsize": 30, "ytick.labelsize": 30,
                                     "xtick.labelsize": 30})
    # Generate a color palette with Seaborn.color_palette()
    pal = sns.cubehelix_palette(len(wanted_interactions), rot=-.25, light=.7)

    # Create a FacetGrid for the Ridge Plot
    g = sns.FacetGrid(filtered_data, row="interaction", hue="interaction", aspect=10, height=3, palette=pal)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    # Remove "label=" prefix from titles
    g.set_titles(row_template="{row_name}", size=30)
    # Map the Ridge Plot using sns.kdeplot()
    g.map(sns.kdeplot, "delta", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "delta", clip_on=False, color="w", lw=2, bw_adjust=.5)

    # Add a horizontal line at y=0
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    plt.subplots_adjust(hspace=0.3)
    # Show the plot
    plt.show()
    g.savefig('RidgePlot.pdf')
    return


# read python dict back from the file
file_path3 = r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\WPTcells results\legRetLists_SELP_WP.pkl"
file_path4 = r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\GPTcells results\legRetLists_SELP_GP.pkl"
file_path5 = r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\WPTcells results\DSA_Tables_SELP_WP.pkl"
file_path6 = r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\GPTcells results\DSA_Tables_SELP_GP.pkl"
path_sig_gp = r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\GPTcells results\GP_Sig_Rec.pkl"
path_sig_wp = r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\WPTcells results\WP_Sig_Rec.pkl"
# create_RidgePlot(r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\merge_all_T.csv",
#                  ['from 0_Neutrophils to CD4', 'from 5_DC(moDC) to CD4', 'from 0_Neutrophils to 2_B_cells',
#                   'from Tregs to CD4', 'from Tregs to 4_Macrophages(M2_like)', 'from 7_Macrophages(TAM) to CD4'])

# merge_all_func(file_path3, file_path4, file_path5, file_path6, path_sig_wp, path_sig_gp)
# print(ligand_list(file_path3, file_path4, '8'))
# print(merge_DSA(dict5['1']['4'], dict6['1']['4']))
# merged = merge_lists(dict3['1']['4'], dict4['1']['4'], dict5['1']['4'], dict6['1']['4'], dict_sig_wp['1']['4'], dict_sig_gp['1']['4'])
# print(merged)
# merge_all_func(file_path3, file_path4, file_path5, file_path6, path_sig_wp, path_sig_gp)
# print(calc_all_p_values(file_path3, file_path4, file_path5, file_path6,
#                             r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\merge_all_T_1.csv", path_sig_wp, path_sig_gp))
# print(create_RidgePlot(
#         ['7to2_sig.csv', '5to2_sig.csv', '7to6_sig.csv', '4to2_sig.csv', '7to0_sig.csv', '4to6_sig.csv', '2to6_sig.csv', '3to4_sig.csv', '5to6_sig.csv', '4to1_sig.csv']))
print(create_diverging_dotplot_1(r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\merge_all_T.csv",
                                 'from Tregs to CD4'))
# print(permutations_func(merged, r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\merge_all.csv", 0.5))
# print(merge_lists(dict2['2']['8'], dict3['2']['8'], dict4['2']['8'], dict5['2']['8']))
# print(merge_DSA(dict4['1']['7'], dict5['1']['7']))
# print(calc_all_DSA_diff(file_path3, file_path4))
# file_path3 = r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\outputObj\DSA_Graphs_SELP.pkl"
# pkl_file3 = open(file_path3, 'rb')
# mydict3 = pickle.load(pkl_file3)
# pkl_file3.close()
# sig_path = r"C:\Users\ofrik\Desktop\Tutorial\interFlow\interFLOW\sig_rec_GP.pkl"
# sig_file = open(sig_path, 'rb')
# dict_sig = pickle.load(sig_file)
# sig_file.close()
# print(dict_sig['1'])
