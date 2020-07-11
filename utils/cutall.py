import os
import json
import thulac
import tqdm

cutter = thulac.thulac(seg_only=True)
frequency = {}

input_path = "/data/disk2/private/zhx/scm/data/all/split"
output_path = "/data/disk2/private/zhx/scm/data/all_cut/split"


def cut(s):
    arr = list(cutter.fast_cut(s))
    for a in range(0, len(arr)):
        arr[a] = arr[a][0]
    for word in arr:
        if not (word in frequency):
            frequency[word] = 0
        frequency[word] += 1
    return arr


if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)
    for filename in tqdm.tqdm(os.listdir(input_path)):
        print(os.path.join(input_path, filename))
        data = []

        inf = open(os.path.join(input_path, filename), "r", encoding="utf8")
        ouf = open(os.path.join(output_path, filename), "w", encoding="utf8")

        for line in inf:
            x = json.loads(line)
            for name in ['QW', 'SBDSR', 'SSJL', 'SS', 'LY', 'JG', 'WB', 'FZ']:
                x[name] = cut(x[name].replace("\n", ""))

            print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=ouf)
