import pandas as pd

def merge_smiles_properties(protac_file, protac_db_file, output_file):
    # 读取两个 CSV 文件
    protac_df = pd.read_csv(protac_file)
    protac_db_df = pd.read_csv(protac_db_file)

    # 确保列名一致以便后续操作
    protac_df.rename(columns={"Smiles": "smiles_canonical"}, inplace=True)

    # 去重：确保 smiles_canonical 在 protac.csv 中是唯一的
    protac_df = protac_df.drop_duplicates(subset="smiles_canonical")

    # 合并数据，保留 protacDB_smiles.csv 中的 id_protac
    merged_df = protac_db_df.merge(
        protac_df,
        on="smiles_canonical",
        how="inner"  # 只保留两者匹配的部分
    )

    # 筛选需要的列，包括 id_protac
    selected_columns = ["id_protac", "smiles_canonical", "Molecular Weight",
                        "Hydrogen Bond Acceptor Count", "Hydrogen Bond Donor Count",
                        "Rotatable Bond Count", "XLogP3"]
    result_df = merged_df[selected_columns]

    # 保存结果到新的 CSV 文件
    result_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}, with {len(result_df)} rows.")

# 输入和输出文件路径
protac_csv = "protac.csv"
protac_db_csv = "preprocess/protacDB_smiles.csv"
output_csv = "protacDB_filtered.csv"

# 运行合并操作
merge_smiles_properties(protac_csv, protac_db_csv, output_csv)
