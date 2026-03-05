import os
import tqdm
from rdkit import Chem
from openbabel import openbabel
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

openbabel.obErrorLog.StopLogging()


def xyz_to_rdkit_mol(xyz_file):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "sdf")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, xyz_file)
    sdf_file = xyz_file.replace('.xyz', '.sdf')
    obConversion.WriteFile(mol, sdf_file)
    sp = Chem.SDMolSupplier(sdf_file)
    return sp[0]


gen_smi_path = '../results/protacs_mc_5'
n = 100

valid_cnt = 0
unique_cnt = 0
total_cnt = 0
recovery_cnt = 0
invalid_input_cnt = 0


def is_valid(xyz_file):
    mol_gene = xyz_to_rdkit_mol(xyz_file)
    try:
        if mol_gene is None:
            return False
        smi = Chem.MolToSmiles(mol_gene)
        # if '.' in smi:
        #     return False
        # Chem.SanitizeMol(mol_gene, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except Exception:
        return False
    return True


gen_smi_dirs = os.listdir(gen_smi_path)
for files in tqdm.tqdm(gen_smi_dirs):
    try:
        # true_xyz_file = f'../datasets-3.0/SDF/{files}_protac.sdf'
        true_xyz_file = f'{gen_smi_path}/{files}/true_.xyz'
        mol = xyz_to_rdkit_mol(true_xyz_file)
        # mol = Chem.MolFromMolFile(true_xyz_file)
        true_smi = Chem.MolToSmiles(mol)
    except Exception as e:
        invalid_input_cnt += 1
        continue
    smi_group = []
    for i in range(n):
        total_cnt += 1
        xyz_file = f'{gen_smi_path}/{files}/{i}_.xyz'
        valid = is_valid(xyz_file)
        if valid:
            valid_cnt += 1
            mol_gene = xyz_to_rdkit_mol(xyz_file)
            smi_group.append(Chem.MolToSmiles(mol_gene))

    if true_smi in smi_group:
        recovery_cnt += 1
    unique_cnt += len(list(set(smi_group)))

validity = valid_cnt / total_cnt * 100
uniqueness = unique_cnt / total_cnt * 100
recovery = recovery_cnt / len(gen_smi_dirs) * 100

print(invalid_input_cnt)
print(gen_smi_path)
print(f'Validity: {validity:.2f}%')
print(f'Uniqueness: {uniqueness:.2f}%')
print(f'Recovery: {recovery:.2f}%')


