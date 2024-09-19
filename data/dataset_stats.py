import os
import numpy as np
from datasets import Dataset, concatenate_datasets

base_dir = "/share/cp/temp/musicmed/metadata_test_32k/"
arrow_files = [os.path.join(base_dir, fn) for fn in os.listdir(base_dir) if fn.endswith(".arrow") and fn.startswith("data")]
ds = concatenate_datasets([Dataset.from_file(arrow_file) for arrow_file in arrow_files])
i = np.unique(np.concatenate(np.char.split(ds["INSTR"], ",")))
i = i[i != ""]
g = np.unique(np.concatenate(np.char.split(ds["GENRE"], ",")))
g = g[g != ""]
i_oh = np.zeros((10459, 41), dtype=int)
g_oh = np.zeros((10459, 95), dtype=bool)
i_oh = np.zeros((10459, 41), dtype=bool)
inst = np.char.split(ds["INSTR"], ",")
genre = np.char.split(ds["GENRE"], ",")
for index in range(len(ds)):
    instruments = inst[index]
    gg = genre[index]
    for instr in instruments:
        i_oh[index][instr == i] = 1
    for gen in gg:
        g_oh[index][gen == g] = 1