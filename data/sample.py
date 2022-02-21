import random


def gen_sample(input_file, output_file, nums):
    with open(input_file) as fp:
        input_lines = fp.readlines()
    sample_train = random.sample(input_lines, nums)
    #sample_train = [i + "\n" for i in sample_train]
    with open(output_file, "w") as fp:
        fp.writelines(sample_train)
    print("save in {}".format(output_file))

gen_sample("trainset1M.txt", "trainset30w.txt", 300000)
gen_sample("validset.txt", "validset5000.txt", 5000)
gen_sample("testset.txt", "testset5000.txt", 5000)