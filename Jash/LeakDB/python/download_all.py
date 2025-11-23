from water_benchmark_hub.leakdb import LeakDB
import os, shutil

db = LeakDB()

missing_scenarios = []

for i in range(980, 1000):
    try:
        data = db.load_data(scenarios_id=[i], use_net1=False, download_dir="D:\LeakDB_full_data")
    
    except (FileNotFoundError, RuntimeError) as e:
        source  = f"D:\LeakDB_full_data\Hanoi\Scenario-{i}\Scenario-{i}"
        target  = f"D:\LeakDB_full_data\Hanoi\Scenario-{i}"
        
        files = os.listdir(source)
        
        for file in files:
            shutil.move(os.path.join(source, file), target)
        
        print(f"Scenario-{i} moved")
        
    except Exception as e:
        print(f"Scenario-{i} cannot be downloaded.")
        missing_scenarios.append(i)