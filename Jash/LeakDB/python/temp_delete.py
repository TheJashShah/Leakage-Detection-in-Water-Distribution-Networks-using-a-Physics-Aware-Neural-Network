import os

missing = [204, 207, 223, 228, 249, 255, 267, 268, 275, 290, 300, 305, 306, 308, 312, 314, 320, 321, 322, 330, 332, 340, 341, 348] 
path  = "D:\LeakDB_full_data\Scada_Data_Full"

for i in missing:
    os.remove(os.path.join(path, f"Hanoi_ID={i}.epytflow_scada_data"))
    print(f"Scenario-{i} [incomplete] removed.")