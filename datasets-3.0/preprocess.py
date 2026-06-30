import os
import torch
import random
import pandas as pd
from tqdm import tqdm
from data_utils import sdf2nx, get_map_ids_from_nx
# from utils.const import NUMBER_OF_ATOM_TYPES, ATOM2IDX, CHARGES
ATOM2IDX = {'C': 0, 'O': 1, 'N': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
IDX2ATOM = {0: 'C', 1: 'O', 2: 'N', 3: 'F', 4: 'S', 5: 'Cl', 6: 'Br', 7: 'I', 8: 'P'}
CHARGES = {'C': 6, 'O': 8, 'N': 7, 'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'P': 15}
NUMBER_OF_ATOM_TYPES = len(ATOM2IDX)


df_data = pd.read_csv('dataset_protac.csv')
ids = list(set([_.split('_')[0] for _ in os.listdir('SDF')]))
train_sets = []
for i, id_i in enumerate(tqdm(ids)):

    G = sdf2nx(f'SDF/{id_i}_protac.sdf')
    G_linker = sdf2nx(f'SDF/{id_i}_linker.sdf')

    maps, anchors = get_map_ids_from_nx(G, G_linker)
    # 5925 -> 5254
    if len(maps) == 1:
        n = len(G.nodes)
        n0 = G.nodes
        n1 = maps[0]  # linker
        n2 = list(set(n0) - set(n1))  # ligand
        positions = []
        one_hot = []
        charges = []
        in_anchors = []
        fragment_mask = []
        linker_mask = []

        flag = False

        for ligand_atom in n2:
            positions.append(G.nodes[ligand_atom]['positions'])
            fragment_mask.append(1.)
            linker_mask.append(0.)

            tmp = [0.] * NUMBER_OF_ATOM_TYPES
            try:
                tmp[ATOM2IDX[G.nodes[ligand_atom]['element']]] = 1.
            except KeyError:
                flag = True
                continue

            one_hot.append(tmp)
            charges.append(CHARGES[G.nodes[ligand_atom]['element']])
            if ligand_atom in anchors[0]:
                in_anchors.append(1.)
            else:
                in_anchors.append(0.)
        if flag:
            continue

        for linker_atom in n1:
            positions.append(G.nodes[linker_atom]['positions'])
            fragment_mask.append(0.)
            linker_mask.append(1.)

            tmp = [0.] * NUMBER_OF_ATOM_TYPES
            tmp[ATOM2IDX[G.nodes[linker_atom]['element']]] = 1.
            one_hot.append(tmp)
            charges.append(CHARGES[G.nodes[linker_atom]['element']])

        train_sets.append({
            'uuid': id_i,
            'name': df_data[df_data['Compound ID'] == int(id_i)]['Smiles_protacs'],
            'positions': torch.tensor(positions),
            'one_hot': torch.tensor(one_hot, dtype=torch.int),
            'charges': torch.tensor(charges),
            'anchors': torch.tensor(in_anchors),
            'fragment_mask': torch.tensor(fragment_mask, dtype=torch.int),
            'linker_mask': torch.tensor(linker_mask),
            'num_atoms': n,
            # properties
            'molecular_weight': df_data[df_data['Compound ID'] == int(id_i)]['Molecular Weight'].iloc[0],
            'XLogP3': df_data[df_data['Compound ID'] == int(id_i)]['XLogP3'].iloc[0],
            'heavy_atom_cnt': df_data[df_data['Compound ID'] == int(id_i)]['Heavy Atom Count'].iloc[0],
            'H_bond_acceptor_cnt': df_data[df_data['Compound ID'] == int(id_i)]['Hydrogen Bond Acceptor Count'].iloc[0],
            'H_bond_donor_cnt': df_data[df_data['Compound ID'] == int(id_i)]['Hydrogen Bond Donor Count'].iloc[0],
            'ring_cnt': df_data[df_data['Compound ID'] == int(id_i)]['Ring Count'].iloc[0],
            'rotatable_bond_cnt': df_data[df_data['Compound ID'] == int(id_i)]['Rotatable Bond Count'].iloc[0],
            'topological_polar_surface_area': df_data[df_data['Compound ID'] == int(id_i)]['Topological Polar Surface Area'].iloc[0]
        })

random.shuffle(train_sets)
total_size = len(train_sets)
print("total size:", total_size)

train_size = int(total_size * 0.8)
val_size = int(total_size * 0.1)

train_data = train_sets[:train_size]
val_data = train_sets[train_size: train_size + val_size]
test_data = train_sets[train_size + val_size:]


TARGET_IDS = {3603, 2067, 2286}

test_cases = [
    sample for sample in train_sets
    if int(sample['uuid']) in TARGET_IDS
]

print("test case size:", len(test_cases))


# torch.save(train_data, './datasets/protacs_train.pt')
# torch.save(val_data, './datasets/protacs_val.pt')
# torch.save(test_data, './datasets/protacs_test.pt')
torch.save(test_cases, './datasets/protacs_test_cases.pt')
