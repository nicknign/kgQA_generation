import os
import sys
import logging
import time
from multiprocessing import Pool
logger = logging.getLogger()
logger.setLevel(logging.INFO)
dir_ = sys.argv[1]
output_dir  = "output_" + dir_

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


cuda_list = '00112233'
cmd_list = []

def os_run(cmd, cuda_no, sleep_time, index):
    cmd = "CUDA_VISIBLE_DEVICES={} ".format(cuda_no) + cmd
    print("sleep {}".format(sleep_time))
    time.sleep(sleep_time)
    print(cmd)
    c = os.system(cmd)
    return c

for r,d,f in os.walk(dir_):
    for file in f:
        if file.endswith(".chkpt"):
            cmd_list.append("/workspace/python3_conda translate_triple.py -src data/test -vocab data/triple_test.pt -model {} -output {}".format(
                os.path.join(r, file),
                os.path.join(output_dir, file+"_out")
            ))

index = 0
for i in range(0, len(cmd_list), len(cuda_list)):
    sub_cmds = cmd_list[i: i + len(cuda_list)]
    args = zip(sub_cmds, cuda_list)
    sleep_time = 0
    pool = Pool(len(cuda_list))
    print("pool start")
    for cmd, cuda_no in args:
        pool.apply_async(os_run, args=(cmd,cuda_no,sleep_time,index))
        #sleep_time += 10
        index += 1
    pool.close()
    pool.join()
    print("pool end")

if os.path.exists("bleu"+dir_):
    os.remove("bleu"+dir_)
if os.path.exists("entity"+dir_):
    os.remove("entity"+dir_)

for r,d,f in os.walk(dir_):
    for file in f:
        os.system("/workspace/python3_conda toChanged.py {}".format(os.path.join(output_dir, file+"_out")))
        print(file)
        os.system("/workspace/python3_conda tools/eval/eval.py -out {} -tgt {} >> {}".format(os.path.join(output_dir, file+"_out_changed"), "data/test-tgt_changed_right", dir_+'_bleu'))
        os.system("/workspace/python3_conda entityCount.py {} >> {}".format(os.path.join(output_dir, file+"_out_changed"), dir_+'_entity'))
