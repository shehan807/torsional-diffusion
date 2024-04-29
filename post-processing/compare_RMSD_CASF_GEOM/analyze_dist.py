import sys 
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.signal import find_peaks
import pandas as pd

def get_rmsd_dist(conformers, smiles, bins=np.linspace(-15,15,101)):
    for num, conf in enumerate(conformers):
        if num == 0: # select the (randomly) first conf for comparison
            compare_mol = conf
            AllChem.EmbedMolecule(compare_mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(compare_mol)
            rmsd_list = []
        rmsd = Chem.rdMolAlign.GetBestRMS(compare_mol, conf) 
        rmsd_list.append(rmsd)
    # convert to histogram 
    rmsd_hist, bins = np.histogram(rmsd_list, bins=bins, density=True)
    mean = np.mean(rmsd_list)
    sigma = np.std(rmsd_list)
    peaks, _ = find_peaks(rmsd_hist, distance=20)
    modes = len(peaks)
    print(f"{len(conformers)} number of conformers; RMSD (u,s,p)=({mean:.2f},{sigma:.2f},{modes}).")
    return rmsd_hist

def get_pca(df, components, features, target, scale=True):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    x = df.loc[:, features].values
    y = df.loc[:, [target]].values

    if scale: 
        x = StandardScaler().fit_transform(x)
    
    pca = PCA(n_components=components)
    principalComponents = pca.fit_transform(x)
    p_df = pd.DataFrame(data=principalComponents, columns=["Component 1", "Component 2"])
    
    finalDf = pd.concat([p_df, df[[target]]], axis = 1)
    return finalDf

def plot_pca(df, target, fig_name="pca.png"):
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=26)
    plt.rc('ytick',labelsize=26)
    plt.rc('grid', c='0.5', ls='-', alpha=0.5, lw=0.5)
    fig = plt.figure(figsize=(22,16))
    ax = fig.add_subplot(1,1,1)
    
    border_width = 1.5; axis_fs=44
    ax.spines['top'].set_linewidth(border_width)
    ax.spines['right'].set_linewidth(border_width)
    ax.spines['bottom'].set_linewidth(border_width)
    ax.spines['left'].set_linewidth(border_width)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction='in', length=8, width=border_width, which='major', top=True, right=True)
    ax.tick_params(direction='in', length=4, width=border_width, which='minor', top=True, right=True)
    ax.set_xlabel(r'Principal Component 1', fontsize=axis_fs)
    ax.set_ylabel(r'Principal Component 2', fontsize=axis_fs)
    
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, color='grey')
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='grey')
    
    datasets = ['DRUGS', 'CASF']
    colors = {"DRUGS":'r',"CASF":'g',"PDBBind":'b'}
    alphas = {"DRUGS":1.0,"CASF":0.9,"PDBBind":0.4}
    for dataset in datasets:
        indicesToKeep = df[target] == dataset
        ax.scatter(df.loc[indicesToKeep, 'Component 1']
                   , df.loc[indicesToKeep, 'Component 2']
                   , c = colors[dataset]
                   , s = 70
                   , alpha=alphas[dataset])
    ax.legend(datasets,loc='upper right', prop={"size":28}, facecolor="white", fancybox=False)
    plt.savefig(fig_name)

with open("../drugs_1order/drugs_steps20.pkl", 'rb') as f:
    drugs_dict = pickle.load(f)
with open("../CASF/casf_final.pkl", 'rb') as f:
    casf_dict = pickle.load(f)

# create dataframe
bins=np.linspace(-15,15,101)
bin_columns = [f'bin_{i}' for i in range(len(bins)-1)] # subtract 1 because these are bin edges
df_columns = bin_columns + ["Dataset","SMILES", "Heavy_Atoms", "Rotatable_Bonds"]
df = pd.DataFrame(columns=df_columns)

err_count = 0
for i, smiles in enumerate(drugs_dict):
    print(f"GEOM-DRUGS: {i} of {len(drugs_dict)}")
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    hvy_atoms = mol.GetNumHeavyAtoms()
    
    try: 
        rmsd_hist = get_rmsd_dist(drugs_dict[smiles], smiles, bins)
    except Exception as e: 
        err_count += 1
        print(f"{err_count}: Couldn't calc RMSD for {smiles} because {e}")
        continue
    
    data = {"Dataset":"DRUGS", "SMILES" : smiles, "Heavy_Atoms":hvy_atoms, "Rotatable_Bonds":rot_bonds}
   
    bin_data = {str(bin_columns[i]):rmsd_hist[i] for i in range(len(bin_columns))}
    data.update(bin_data)
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)
    # print(smiles, new_row) # debug empty concat warning 
err_count = 0 
for conf_num, smiles in enumerate(casf_dict["smile_str"]):
    print(f"{conf_num} of {len(casf_dict['smile_str'])}")
    conformers = casf_dict["conformers"][conf_num]
   
    try: 
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
    except Exception as e:
        err_count += 1
        print(f"{err_count}: Couldn't process {smiles}")
        continue
    rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    hvy_atoms = mol.GetNumHeavyAtoms()
    
    try: 
        rmsd_hist = get_rmsd_dist(conformers, smiles, bins)
    except Exception as e: 
        err_count += 1
        print(f"{err_count}: Couldn't calc RMSD for {smiles} because {e}")
        continue
    
    data = {"Dataset":"CASF", "SMILES" : smiles, "Heavy_Atoms":hvy_atoms, "Rotatable_Bonds":rot_bonds}
    
    bin_data = {str(bin_columns[i]):rmsd_hist[i] for i in range(len(bin_columns))}
    data.update(bin_data)
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)

df.to_pickle("df.pkl")

features = ["Heavy_Atoms", "Rotatable_Bonds"]
target   = "Dataset"
p_df = get_pca(df, 2, features, target, scale=False)
plot_pca(p_df, "Dataset", fig_name="pca_no_rmsd.png")

features = ["Heavy_Atoms", "Rotatable_Bonds"]
target   = "Dataset"
p_df = get_pca(df, 2, features, target, scale=True)
plot_pca(p_df, "Dataset", fig_name="pca_no_rmsd_scale.png")

features = bin_columns # ["Heavy_Atoms", "Rotatable_Bonds"]
target   = "Dataset"
p_df = get_pca(df, 2, features, target, scale=False)
plot_pca(p_df, "Dataset", fig_name="pca_rmsd_noscale.png")

features = bin_columns # ["Heavy_Atoms", "Rotatable_Bonds"]
target   = "Dataset"
p_df = get_pca(df, 2, features, target, scale=True)
plot_pca(p_df, "Dataset", fig_name="pca_rmsd_scale.png")

