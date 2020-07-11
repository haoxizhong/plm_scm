import os
import multiprocessing

input_data_path = "/data/disk2/private/zhx/scm/data/pretrain"
output_data_path = "/data/disk2/private/zhx/scm/data/pretrain_bin"
os.makedirs(output_data_path,exist_ok=True)

num_files = 100
num_process = 5

q = multiprocessing.Queue()
bq = multiprocessing.Queue()

for a in range(1, num_files + 1):
    q.put(a)


def work():
    while True:
        a = q.get(timeout=5)

        os.system("""
        python3 create_instances_fast.py --input_file_prefix %s/%d --output_file %s/%d --vocab_file ../utils/word2id.txt --dupe_factor 1 --max_seq_length 512 --max_predictions_per_seq 80 --random_seed 0
""" % (input_data_path, a, output_data_path, a))

        bq.put(1)


if __name__ == "__main__":
    process_list = []
    for a in range(0, num_process):
        process = multiprocessing.Process(target=work)
        process_list.append(process)
        process_list[-1].start()

    done = 0

    while done < num_files:
        _ = bq.get()

        done += _

        print("%d/%d\r" % (done, num_files), end="")
