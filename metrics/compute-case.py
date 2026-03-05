import os
from rdkit import Chem
import openbabel
from tqdm import tqdm
from rdkit import RDLogger
import pandas as pd

RDLogger.DisableLog('rdApp.*')
openbabel.obErrorLog.StopLogging()

def is_valid(mol_gene):
    try:
        if mol_gene is None:
            return False, None
        smi = Chem.MolToSmiles(mol_gene)
        if '.' in smi:
            return False, None
        Chem.SanitizeMol(mol_gene, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        return False, None
    return True, smi

def xyz_to_rdkit_mol(xyz_file):
    """
    利用 OpenBabel 将 xyz 文件转换为 sdf 文件，再用 RDKit 读取 sdf 文件生成分子对象
    """
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")
    mol = openbabel.OBMol()
    # 读取临时保存的 linker xyz 文件
    obConversion.ReadFile(mol, xyz_file)
    sdf_file = xyz_file.replace('.xyz', '.sdf')
    obConversion.WriteFile(mol, sdf_file)
    return Chem.MolFromMolFile(sdf_file)

def extract_linker(xyz_file, skip_atoms):
    """
    从 xyz 文件中提取 linker 部分：
    - xyz 文件格式：第一行为原子总数，第二行为注释，后续为原子坐标行。
    - skip_atoms 为固定原子数（非 linker 部分），因此 linker 从第 (skip_atoms + 2) 行开始
    """
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    # 提取从第 (skip_atoms + 2) 行开始的原子坐标行
    linker_atoms = lines[2 + skip_atoms:]
    linker_count = len(linker_atoms)
    # 重新组装为 xyz 格式：第一行为 linker 原子数，第二行为注释
    new_xyz = f"{linker_count}\nLinker extracted from {os.path.basename(xyz_file)}\n" + "".join(linker_atoms)
    return new_xyz

def validate_linkers(main_dir):
    """
    遍历主目录下以纯数字命名的子文件夹，依次验证 linker 的有效性。
    返回每个子文件夹的 Validity。
    """
    result = {}
    for subfolder in tqdm(os.listdir(main_dir)):
        total_cnt = 0
        valid_cnt = 0
        subfolder_path = os.path.join(main_dir, subfolder)
        if os.path.isdir(subfolder_path):
            frag_file = os.path.join(subfolder_path, "frag_.xyz")
            true_file = os.path.join(subfolder_path, "true_.xyz")
            with open(frag_file, 'r') as f:
                fixed_atoms = int(f.readline().strip())

            true_linker_str = extract_linker(true_file, fixed_atoms)
            true_temp_file = os.path.join(subfolder_path, "temp_true_linker.xyz")
            with open(true_temp_file, 'w') as f:
                f.write(true_linker_str)
            true_mol = xyz_to_rdkit_mol(true_temp_file)
            true_flag, true_smi = is_valid(true_mol)
            # if not true_flag:
            #     continue

            smi_group = []
            for i in range(100):
                total_cnt += 1
                gen_file = os.path.join(subfolder_path, f"{i}_.xyz")
                xyz_to_rdkit_mol(gen_file)
                linker_str = extract_linker(gen_file, fixed_atoms)
                temp_file = os.path.join(subfolder_path, f"temp_{i}_linker.xyz")
                with open(temp_file, 'w') as f:
                    f.write(linker_str)
                gen_mol = xyz_to_rdkit_mol(temp_file)
                flag, smi = is_valid(gen_mol)
                if flag:
                    smi_group.append(smi)

            valid_cnt = len(smi_group)
            validity = valid_cnt / total_cnt * 100 if total_cnt > 0 else 0
            result[subfolder] = validity

    return result


if __name__ == "__main__":
    # 初始化一个空的 dataframe
    df = pd.DataFrame()

    for size in range(10, 18):  
        main_directory = f"protacs_mc_la_case-2/size_{size}"
        validity_results = validate_linkers(main_directory)

        for subfolder_name, validity in validity_results.items():
            df.at[subfolder_name, f"size={size}"] = validity

    # 保存到本地
    df.to_csv("linker_validity_results.csv")
    print("结果已保存到 linker_validity_results.csv")
