import os
import csv
import re
from rdkit import Chem

id2linkerSize = {'1319': 8, '2097': 7, '3271': 16, '3272': 10, '3273': 11}

def extract_linker_smiles(sdf_path, linker_size):
    """
    从SDF文件中提取后8个原子组成的linker的SMILES。
    :param sdf_path: SDF文件路径
    :return: Linker的SMILES字符串，若失败则返回None
    """
    try:
        supplier = Chem.SDMolSupplier(sdf_path)
        mol = next(supplier)  # 假设文件中只有一个分子
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            # 获取后8个原子
            num_atoms = mol.GetNumAtoms()
            linker_atom_indices = list(range(num_atoms - linker_size, num_atoms))  # 后8个原子的索引

            # 提取子结构
            linker = Chem.RWMol(mol)
            atoms_to_remove = [atom.GetIdx() for atom in linker.GetAtoms() if atom.GetIdx() not in linker_atom_indices]
            atoms_to_remove.sort(reverse=True)  # 从后往前删除，避免索引变化
            for atom_idx in atoms_to_remove:
                linker.RemoveAtom(atom_idx)

            # 生成SMILES
            linker_smiles = Chem.MolToSmiles(linker)
            return linker_smiles, smiles
        else:
            return None
    except Exception as e:
        print(f"处理文件 {sdf_path} 时出错: {e}")
        return None

def simplify_smiles(smiles):
    """
    去除SMILES中包裹单字母原子符号的 []。
    :param smiles: 输入的SMILES字符串
    :return: 简化后的SMILES字符串
    """
    simplified = re.sub(r'\[([CNO])\]', r'\1', smiles)
    return simplified

def process_folder_to_csv(folder_path, output_csv, linker_size):
    """
    处理文件夹中的所有SDF文件，并将结果保存到CSV文件中。
    :param folder_path: 包含SDF文件的文件夹路径
    :param output_csv: 输出的CSV文件路径
    """
    # 准备CSV文件
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['ID', 'Linker_SMILES', 'SMILES'])  # 写入表头

        # 遍历文件夹中的SDF文件
        for i in range(100):  # 0到 99
            sdf_file = os.path.join(folder_path, f"{i}_.sdf")
            if os.path.exists(sdf_file):
                out = extract_linker_smiles(sdf_file, linker_size)
                if out is not None:
                    linker_smiles, smiles = out
                else:
                    smiles = None
                if smiles:
                    simplified_linker_smiles = simplify_smiles(linker_smiles)  # 去除多余的 []
                    simplified_smiles =  simplify_smiles(smiles)
                    writer.writerow([i, simplified_linker_smiles, simplified_smiles])  # 写入ID和SMILES
                    print(f"处理成功: {sdf_file} -> {simplified_smiles}")
                else:
                    print(f"未提取到SMILES: {sdf_file}")
            else:
                print(f"文件不存在: {sdf_file}")

# 示例调用
for file_id, linker_size in id2linkerSize.items():
    folder_path = f"output/{file_id}"  # 替换为你的文件夹路径
    output_csv = f"output/output_linkers_{file_id}_simple.csv"  # 输出的CSV文件路径
    process_folder_to_csv(folder_path, output_csv, linker_size)