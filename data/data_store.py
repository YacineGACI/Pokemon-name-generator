import torch 

class DataStore:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None


    def read_raw_data(self):
        with open(self.filepath, 'r') as f:
            self.raw_data = [pokemon.lower()for pokemon in f.read().strip().split('\n')]

    