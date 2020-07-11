import json


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_english(uchar):
    return 32 <= ord(uchar) <= 127


min_frequency = 10

data = json.load(open("/data/disk2/private/zhx/scm/data/all/word_count.json", "r"))

res = {
    "[PAD]": 0,
}

for a in range(1, 100):
    res["[unused%d]" % a] = a

res["[UNK]"] = 100
res["[CLS]"] = 101
res["[SEP]"] = 102
res["[MASK]"] = 103

cnt = 103
for word in data.keys():
    if word == " ":
        continue
    if word == "\n":
        if data[word] < min_frequency:
            continue
    cnt += 1
    res[word] = cnt

f = open("/data/disk2/private/zhx/scm/data/all/word2id.txt", "w")
for x in res.keys():
    print(x, file=f)

f.close()

json.dump(res, open("/data/disk2/private/zhx/scm/data/all/word2id.json", "w"), indent=2, ensure_ascii=False)
