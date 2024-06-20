import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader

D_VOCAB = 64

# Util functions

def parse_data(data):
    def parse_inputs(row):
        """Given a row, returns the input pattern."""
        lst = row[1]
        pattern = ""

        # Unpack the list into a string of values
        for i in lst:
            pattern += f"{i} "

        # Add the remaining part
        pattern += "[SEP] "
        pattern += "[MASK] " * len(lst)   
        # trim the space at the end
        pattern = pattern.strip()
        return pattern

    def parse_outputs(row):
        """Given a row, returns the output pattern."""
        lst = row[0]
        pattern = ""

        # Unpack the list into a string of values
        for i in lst:
            pattern += f"{i} "
        pattern = pattern.strip()
        return pattern

    """Given a dataframe, returns the input and output patterns."""
    data["text"] = data.apply(parse_inputs, axis=1)
    data["labels"] = data.apply(parse_outputs, axis=1)
    return data.loc[:, ["text", "labels"]]


class SwapDataset(Dataset):

    def sorted_list_random_numbers(self,length):
        """Generates a list of random numbers of a given length and sorts them."""
        return sorted([random.randint(0, D_VOCAB) for i in range(length)], reverse=True)

    def swap_two_numbers(self,sorted_list):
        """Given a sorted list, swaps two numbers in the list."""
        ls = sorted_list.copy()
        idx = random.randint(0,len(ls)-2)
        ls[idx], ls[idx+1] = ls[idx+1], ls[idx]
        return ls

    def generate_swap_dataset(self, length=4, num_samples=2):
        sorted_lst = [self.sorted_list_random_numbers(length) for i in range(num_samples)]
        df = pd.DataFrame()
        df["sorted"] = sorted_lst
        df["swapped"] = [self.swap_two_numbers(lst) for lst in sorted_lst]
        data = parse_data(df)
        return data
    
    def __init__(self, num_samples=2000, length=4):
        self.data = self.generate_swap_dataset(num_samples=num_samples, length=length)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data["text"][idx], self.data["labels"][idx]

class DisplaceDataset(Dataset):

    def __init__(self, length=4, num_samples=2000):
        print("hELLO")
        self.data = self.generate_displace_dataset(length=length, num_samples=num_samples)
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data["text"][idx], self.data["labels"][idx]
    
    def sorted_list_random_numbers(self, length):
        return sorted([random.randint(0, D_VOCAB) for i in range(length)], reverse=True)

    def displace(self, lst):
        """Given a list, displaces item at index i to index j in the list."""
        i, j = 0,0
        while abs(i - j) < 1:
            i = random.randint(0, len(lst)-1)
            j = random.randint(0, len(lst)-1)
        if i > j:
            new_lst = lst[:j]
            new_lst.append(lst[i])
            new_lst += lst[j:i]
            new_lst += lst[i+1:]
        else: # j < i
            new_lst = lst[:i]
            new_lst += lst[i+1:j]
            new_lst.append(lst[i])
            new_lst += lst[j:]
        return new_lst

    def generate_displace_dataset(self, length=4, num_samples = 2000):
        sorted_lst = [self.sorted_list_random_numbers(length) for i in range(num_samples)]
        df = pd.DataFrame()
        df["sorted"] = sorted_lst
        df["displaced"] = [self.displace(lst) for lst in sorted_lst]
        data = parse_data(df)
        return data


class TopKDataset(Dataset):
    def generate_input(self, length, num_range):
        """Generates a list of random numbers of a given length and sorts them."""
        return [random.randint(0, num_range) for i in range(length)] 

    def __init__(self, num_samples, length, K, num_range=64, random_seed=None):
        if random_seed:
            random.seed(random_seed)
        input_lists = [self.generate_input(length, num_range) for i in range(num_samples)]
        sorted_input_lists = [sorted(lst, reverse=True) for lst in input_lists]
        top_k = [sorted_input_lists[i][:K] for i in range(num_samples)]
        text = [" ".join([str(i) for i in lst]) for lst in input_lists]
        labels = [" ".join([str(i) for i in lst]) for lst in top_k]
        for i in range(num_samples):
            text[i] += ' [SEP]'
            text[i] += ' [MASK]' * K
        self.data = pd.DataFrame({"text": text, "labels": labels})
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.data["text"][idx], self.data["labels"][idx])

class ListSortDataset(Dataset):
    def generate_input(self, length, num_range):
        """Generates a list of random numbers of a given length and sorts them."""
        return [random.randint(0, num_range) for i in range(length)] 

    def __init__(self, num_samples, list_length, num_range=64, random_seed=None):
        if random_seed:
            random.seed(random_seed)
        input_lists = [self.generate_input(list_length, num_range) for i in range(num_samples)]
        sorted_lists = [sorted(lst, reverse=True) for lst in input_lists]
        text = [" ".join([str(i) for i in lst]) for lst in input_lists]
        labels = [" ".join([str(i) for i in lst]) for lst in sorted_lists]
        for i in range(num_samples):
            text[i] += ' [SEP]'
            text[i] += ' [MASK]' * list_length
        self.data = pd.DataFrame({"text": text, "labels": labels})
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (self.data["text"][idx], self.data["labels"][idx])
    
if __name__ == "__main__":
    displace_dataset = DisplaceDataset(length=6, num_samples=100)
    data_loader = DataLoader(displace_dataset, batch_size=4, shuffle=True)
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        print("Batch: ", batch_idx)
        print(inputs)
        print(labels)
    
    
