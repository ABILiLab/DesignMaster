import os
import glob
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from scipy.stats import wasserstein_distance
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def compute_properties(mol):
    """
    根据给定分子计算各属性
    """
    props = {}
    props["Molecular Weight"] = Descriptors.MolWt(mol)
    props["Hydrogen Bond Acceptor Count"] = Lipinski.NumHAcceptors(mol)
    props["Hydrogen Bond Donor Count"] = Lipinski.NumHDonors(mol)
    props["Rotatable Bond Count"] = Lipinski.NumRotatableBonds(mol)
    props["XLogP3"] = Crippen.MolLogP(mol)  # 近似计算 XLogP3
    props["Topological Polar Surface Area"] = Descriptors.TPSA(mol)
    props["Heavy Atom Count"] = mol.GetNumHeavyAtoms()
    return props

def load_generated_properties(samples_dir):
    """
    遍历 samples 文件夹下所有子文件夹，读取所有固定命名格式的 .sdf 文件计算分子属性
    """
    generated_properties = {
        "Molecular Weight": [],
        "Hydrogen Bond Acceptor Count": [],
        "Hydrogen Bond Donor Count": [],
        "Rotatable Bond Count": [],
        "XLogP3": [],
        'Heavy Atom Count': [],
        'Topological Polar Surface Area':[]
    }

    for subfolder in tqdm(os.listdir(samples_dir)):
        subfolder_path = os.path.join(samples_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # 固定命名格式：0_.sdf 到 99_.sdf
            for i in range(100):
                sdf_file = os.path.join(subfolder_path, f"temp_{i}_linker.sdf")
                if not os.path.exists(sdf_file):
                    # print(f"文件不存在：{sdf_file}")
                    continue
                try:   
                    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
                except Exception:
                    continue
                if not suppl or len(suppl) == 0 or suppl[0] is None:
                    # print(f"无法读取分子：{sdf_file}")
                    continue
                mol = suppl[0]
                try:
                    props = compute_properties(mol)
                except Exception as e:
                    print(f"计算 {sdf_file} 属性时出错: {e}")
                    continue
                for key in generated_properties:
                    generated_properties[key].append(props[key])
    return generated_properties

def compare_distributions(real_csv, generated_properties):
    """
    读取真实数据 CSV 后，与生成分子的属性分布进行对比，使用 Wasserstein 距离度量分布差异
    """
    real_df = pd.read_csv(real_csv)
    for prop in generated_properties:
        # 假设 CSV 文件中对应的列名与 generated_properties 的 key 一致
        if prop not in real_df.columns:
            print(f"真实数据中缺少属性：{prop}")
            continue
        real_vals = real_df[prop].dropna().tolist()
        gen_vals = generated_properties[prop]
        # 计算 Wasserstein 距离
        distance = wasserstein_distance(gen_vals, real_vals)
        print(f"属性：{prop} 的 Wasserstein 距离为：{distance}")
def visualize_all_distributions(real_csv, generated_properties):
    """
    将真实数据和生成数据中每个属性的分布情况绘制在一张大图中，
    每个属性占用一个子图（这里设置为5行1列），以便直观比较。
    """
    real_df = pd.read_csv(real_csv)
    properties = list(generated_properties.keys())
    n_props = len(properties)
    
    # 创建一个包含 n_props 个子图的画布
    fig, axes = plt.subplots(n_props, 1, figsize=(10, 4 * n_props))
    
    # 如果只有一个子图，将 axes 转换成列表
    if n_props == 1:
        axes = [axes]
    
    for i, prop in enumerate(properties):
        ax = axes[i]
        if prop not in real_df.columns:
            print(f"真实数据中缺少属性：{prop}")
            continue

        real_vals = real_df[prop].dropna().tolist()
        gen_vals = generated_properties[prop]
        
        # 绘制真实数据和生成数据的直方图以及核密度估计曲线
        sns.histplot(real_vals, color='blue', label='Real', kde=True, stat='density', bins=30, alpha=0.6, ax=ax)
        sns.histplot(gen_vals, color='orange', label='Generated', kde=True, stat='density', bins=30, alpha=0.6, ax=ax)
        
        ax.set_title(f"{prop} Distribution")
        ax.set_xlabel(prop)
        ax.set_ylabel("Density")
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join('prop_vis.png',))
        
if __name__ == "__main__":
    # 真实数据 CSV 文件路径（请根据实际情况修改）
    real_csv_path = "datasets-3.0/PROTAC-DB-3.0/linker.csv"
    # 生成数据所在文件夹路径
    samples_dir = "protacs_mc_cfg"
    # samples_dir = "../DiffPROTACs-main/DiffPROTACs-main/protacs_baseline"

    # 计算生成分子的属性
    generated_properties = load_generated_properties(samples_dir)
    # 对比生成分子和真实分子的属性分布
    compare_distributions(real_csv_path, generated_properties)
    visualize_all_distributions(real_csv_path, generated_properties)
