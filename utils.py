import torch 
import random, time, math

all_letters = "abcdefghijklmnopqrstuvwxyz .-é'1234567890’"
num_letters = len(all_letters) + 1



def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def letter2tensor(letter):
    tensor = torch.zeros(1, num_letters)
    letter_index = all_letters.find(letter)
    tensor[0][letter_index] = 1
    return tensor


def pokemon2tensor(pokemon):
    tensor = torch.zeros(len(pokemon), 1, num_letters)
    for i in range(len(pokemon)):
        letter_index = all_letters.find(pokemon[i])
        tensor[i][0][letter_index] = 1
    return tensor


def inputTensor(pokemons):
    pokemon = randomChoice(pokemons)
    return pokemon, pokemon2tensor(pokemon)


def targetTensor(pokemon):
    letter_indexes = [all_letters.find(pokemon[i]) for i in range(1, len(pokemon))]
    letter_indexes.append(num_letters - 1)
    return torch.LongTensor(letter_indexes)


def randomTrainingExample(pokemons):
    pokemon, input_tensor = inputTensor(pokemons)
    target_tensor = targetTensor(pokemon)
    return pokemon, input_tensor, target_tensor


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

