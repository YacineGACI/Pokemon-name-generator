import torch
import math, random
import utils
import numpy as np


def sample():
    current_letter = utils.randomChoice('abcdefghijklmnopqrstuvwxyz')
    generated_letters = [current_letter]
    hidden = torch.zeros(1, hidden_size)
    while True:
        current_tensor = utils.letter2tensor(current_letter)
        hidden, output = model(current_tensor, hidden)
        output = [math.exp(x) for x in output.data[0].numpy()]
        output = np.cumsum(output)
        prob = random.random()
        index = 0
        while output[index] < prob:
            index += 1
        if index == utils.num_letters - 1:
            break #End of name
        current_letter = utils.all_letters[index]
        generated_letters.append(current_letter)
    return "".join(generated_letters)
        




if __name__ == "__main__":
    model = torch.load("trained_models/pokemon_generator-500000-epochs.pt")
    hidden_size = 200
    model.eval()
    print(sample())