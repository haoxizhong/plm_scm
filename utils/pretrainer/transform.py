import os
import json
import multiprocessing
import random

input_data_path = "/data/disk2/private/zhx/scm/data/all/split"
output_data_path = "/data/disk2/private/zhx/scm/data/data/pretrain"
os.makedirs(output_data_path, exist_ok=True)

q = multiprocessing.Queue()
bq = multiprocessing.Queue()
file_list = []
cnt = 0
for filename in os.listdir(os.path.join(input_data_path)):
    file_list.append(os.path.join(input_data_path, filename))
    cnt += 1

print(cnt)

random.shuffle(file_list)

per_file = 1
cx = 0
for a in range(0, len(file_list), per_file):
    cx += 1
    arr = []
    for b in range(a, min(a + per_file, len(file_list))):
        arr.append(file_list[b])
    q.put((cx, arr))

print(cx)

num_process = 20

split_list = ["。"]

word_to_dic = json.load(open("/data/disk2/private/zhx/scm/data/all/word2id.json", "r"))


def load(c):
    if not (c in word_to_dic.keys()):
        c = "[UNK]"
    return word_to_dic[c]


def transform(id_, file_list):
    f = open(os.path.join(output_data_path, str(id_)), "w")
    for file_name in file_list:
        inf = open(file_name, "r")
        for line in inf:
            x = json.loads(line)
            arr = [[]]
            s = x["QW"]
            s = s.replace(" ", "").replace("\t", "")
            for a in range(0, len(s)):
                if s[a] in split_list:
                    if len(arr[-1]) == 0:
                        continue
                    arr.append([])
                else:
                    arr[-1].append(load(s[a]))

            while len(arr) > 0 and len(arr[-1]) == 0:
                arr = arr[:-1]
            if len(arr) == 0:
                continue
            print(len(arr), end=' ', file=f)
            for a in range(0, len(arr)):
                print(len(arr[a]), end=' ', file=f)
                for b in range(0, len(arr[a])):
                    print(arr[a][b], end=' ', file=f)
            print("", file=f)

    f.close()


def work():
    while True:
        id_, file_list = q.get(timeout=5)

        transform(id_, file_list)

        bq.put(len(file_list))


if __name__ == "__main__":
    process_list = []
    for a in range(0, num_process):
        process = multiprocessing.Process(target=work)
        process_list.append(process)
        process_list[-1].start()

    done = 0

    while done < cnt:
        _ = bq.get()

        done += _

        print("%d/%d\r" % (done, cnt), end="")
