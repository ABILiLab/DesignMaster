import os
import sys
from rdkit import Chem
from utils import disable_rdkit_logging
import subprocess
import tqdm

disable_rdkit_logging()

gen_smi_path = 'protacs_all'
n = 100

valid_cnt = 0
unique_cnt = 0
total_cnt = 0
recovery_cnt = 0


def is_valid(xyz_file, sdf_file):
    try:
        command = ["obabel", xyz_file, "-O", sdf_file]
        _ = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        mol = Chem.SDMolSupplier(sdf_file)[0]
        if mol_gene is None:
            return False
        smi = Chem.MolToSmiles(mol)
        print(smi)
        # smi = Chem.MolToSmiles(mol_gene)
        # smi = mol.write(format="smi")
        # if '.' in smi:
        #     return False
        Chem.SanitizeMol(mol_gene, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        return False

    return True


gen_smi_dirs = os.listdir(gen_smi_path)
for files in tqdm.tqdm(gen_smi_dirs):

    # mol = next(obabel.readfile("xyz", f'{gen_smi_path}/{files}/true_.xyz'))
    # true_smi = mol.write(format="smi")
    
    true_xyz_file = f'{gen_smi_path}/{files}/true_.xyz'
    true_sdf_file = f'{gen_smi_path}/{files}/true_.sdf'
    command = ["obabel", true_xyz_file, "-O", true_sdf_file]
    _ = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mol = Chem.SDMolSupplier(true_sdf_file)[0]
    if mol is None:
        continue
    true_smi = Chem.MolToSmiles(mol)
    # print("***********************************************")
    # print(true_smi)
    smi_group = []
    for i in range(n):
        total_cnt += 1
        xyz_file = f'{gen_smi_path}/{files}/{i}_.xyz'
        sdf_file = f'{gen_smi_path}/{files}/{i}_.sdf'
        valid = is_valid(xyz_file, sdf_file)
        if valid:
            valid_cnt += 1
            mol_gene = Chem.SDMolSupplier(sdf_file)[0]
            smi_group.append(Chem.MolToSmiles(mol_gene))

    if true_smi in smi_group:
        recovery_cnt += 1
    unique_cnt += len(list(set(smi_group)))

validity = valid_cnt / total_cnt * 100
uniqueness = unique_cnt / total_cnt * 100
recovery = recovery_cnt / len(gen_smi_dirs) *100
print(gen_smi_path)
print(f'Validity: {validity:.2f}%')
print(f'Uniqueness: {uniqueness:.2f}%')
print(f'Recovery: {recovery:.2f}%')
