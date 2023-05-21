import random


class Memory:
    def __init__(self, size_max, size_min):
        self.samples = []
        self.max_size = size_max
        self.min_size = size_min

    def add_sample(self, sample):
        self.samples.append(sample)
        if self.present_size() > self.max_size:
            self.samples.pop(0)  

    def get_samples(self, n):
        if self.present_size() < self.min_size:
            return []

        if n > self.present_size():
            return random.sample(self.samples, self.present_size()) 
        else:
            return random.sample(self.samples, n)  
        
    def present_size(self):
        return int(len(self.samples))