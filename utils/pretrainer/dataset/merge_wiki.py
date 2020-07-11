import indexed_dataset
import os

builder = indexed_dataset.IndexedDatasetBuilder('merge.bin')
for filename in os.listdir("/data3/zzy_tmp/output"):
    if filename[-4:] == '.bin':
        builder.merge_file_("/data3/zzy_tmp/output/" + filename[:-4])
builder.finalize("merge.idx")
