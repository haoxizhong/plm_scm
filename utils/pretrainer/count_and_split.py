import os
import json

input_path = "/data/disk2/private/zhx/scm/data/all/data.json"
output_path = "/data/disk2/private/zhx/scm/data/all/split"
key_list = ["TITLE", "QW"]
dic = {}
os.makedirs(output_path, exist_ok=True)
num_file = 100
file_list = []
for a in range(0, num_file):
    file_list.append(open(os.path.join(output_path, str(a)), "w", encoding="utf8"))


def add(x):
    if not (x in dic.keys()):
        dic[x] = 0
    dic[x] += 1


if __name__ == "__main__":

    cnt = 0
    f = open(input_path, "r", encoding="utf8")
    for line in f:
        data = json.loads(line)
        for key in key_list:
            if key in data.keys():
                for a in range(0, len(data[key])):
                    add(data[key][a])

        cnt += 1
        print("%d\r" % cnt, end="")
        print(json.dumps(data, ensure_ascii=False, sort_keys=True), file=file_list[cnt % num_file])

    json.dump(dic, open("/data/disk2/private/zhx/scm/data/all/word_count.json", "w"), indent=2, ensure_ascii=False,
              sort_keys=True)
