import os
path = "D:\LeakDB_full_data\Scada_Data_Full"

from water_benchmark_hub.leakdb import LeakDB
db = LeakDB()

missing = []

for i in [204, 207, 223, 228, 249, 255, 267, 268, 275, 290, 300, 305, 306, 308, 312, 314, 320, 321, 322, 330, 332, 340, 341, 348]:
    
    try:
        data = db.load_scada_data(scenarios_id=[i], use_net1=False, download_dir=path)
        print(f"Scenario={i} downloaded.")
        
    except Exception as e:
        print(f"Scenario={i} could not be downloaded.")
        missing.append(i)
        
with open("missing.txt", "a") as file:
    file.write(f"Missing scenarios are: {missing} \n")