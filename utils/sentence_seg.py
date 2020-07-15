import re
import os
import json
import thulac



path_list = [
    ["/data/disk3/private/zhx/theme/data/scm/origin",
     "/data/disk3/private/wangyuzhong/scm/data/cail2018_cutted"]
]


def sentence_segmentation(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


if __name__ == "__main__":
    arr_len=[]
    for input_path, output_path in path_list:
        os.makedirs(output_path, exist_ok=True)
        for filename in os.listdir(input_path):
            print(os.path.join(input_path, filename))
            data = []

            f = open(os.path.join(input_path, filename), "r", encoding="utf8")

            for line in f:
                x = json.loads(line)
                for name in ["A", "B", "C"]:
                    x[name] = sentence_segmentation(x[name])
                    data.extend(x[name])
                    arr_len.extend(list(map(len,x[name])))

                # data.append(x)

            f = open(os.path.join(output_path, filename), "w", encoding="utf8")
            for x in data:
                print(json.dumps(x, ensure_ascii=False, sort_keys=True), file=f)
            f.close()

    arr_len.sort()
    n=len(arr_len)
    with open("/data/disk3/private/wangyuzhong/scm/data/cail2018_cutted/length_distribution.txt","w") as f:
        for i in range(20):
            f.write("{}% : {}\n".format(i*100/20,arr_len[n*i//20]))
        f.write("max : {}\n".format(arr_len[-1]))

    '''
    json.dump(frequency, open("/data/disk3/private/zhx/theme/data/scm/frequency.txt", "w", encoding="utf8"),
              indent=2,
              ensure_ascii=False)
    '''


