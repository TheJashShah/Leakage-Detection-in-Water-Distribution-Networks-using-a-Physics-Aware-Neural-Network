import os

dir = "D:\LeakDB_Data\Hanoi"

scenario_1 = os.path.join(dir, "Scenario-1")
inp_file_1 = os.path.join(scenario_1, "Hanoi_CMH_Scenario-1.inp")

scenario_600 = os.path.join(dir, "Scenario-600")
inp_file_600 = os.path.join(scenario_1, "Hanoi_CMH_Scenario-600.inp")

with open(inp_file_1, "r") as file:
    lines = file.readlines()
    
root_dir = os.getcwd()
current_dir = os.path.join(root_dir, "LeakDB")
text = os.path.join(current_dir, "text")
graph = os.path.join(text, "inp_1_text.txt")

if not (os.path.exists(os.path.join(current_dir, graph))):
    with open(os.path.join(current_dir, graph), "w") as file:
        file.writelines(lines)
   
graph_1 = []
 
for i in range(48, 82):
    row = lines[i].split(" ")
    row = [x.strip() for x in row if x.strip() != '']
    row.pop()
    graph_1.append(row)

'''
INP FILE CONVENTION
PIPE_ID NODE1 NODE2 LENGTH DIAMETER ROUGHNESS MINOR LOSS STATUS
'''

dataframe_1 = {}

pipe_id = []
node1 = []
node2 = []
pipe_length = []
pipe_diameter = []
pipe_roughness = []
pipe_minor_loss = []
pipe_status = []

for row in graph_1:
    pipe_id.append(row[0])
    node1.append(row[1])
    node2.append(row[2])
    pipe_length.append(row[3])
    pipe_diameter.append(row[4])
    pipe_roughness.append(row[5])
    pipe_minor_loss.append(row[6])
    pipe_status.append(row[7])
    
dataframe_1["pipe_id"] = pipe_id
dataframe_1["node1"] = node1
dataframe_1["node2"] = node2
dataframe_1["pipe_length"] = pipe_length
dataframe_1["pipe_diameter"] = pipe_diameter
dataframe_1["pipe_roughness"] = pipe_roughness
dataframe_1["pipe_minor_loss"] = pipe_minor_loss
dataframe_1["pipe_status"] = pipe_status

import pandas as pd

data_1 = pd.DataFrame(dataframe_1)
data_1.to_csv(os.path.join(current_dir, "graph_1.csv"))