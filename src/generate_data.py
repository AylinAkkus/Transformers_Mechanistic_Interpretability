import numpy as np

'''
Write a dataset to a file
Each line in the file is a sample
Each sample is three integers separated by a space
The first and second integers are randomly generated different integers 
The third integer is the larger one of the first two integers
'''
def generate_data(file_path, num_samples):
    with open(file_path, 'w') as f:
        for i in range(num_samples):
            a = np.random.randint(0, 100)
            b = np.random.randint(0, 100)
            c = max(a, b)
            f.write(f"{a} {b} {c}\n")


if __name__ == "__main__":
    generate_data("huggingface/data/train.txt", 100)
