import torch 
import torch.nn as nn 

from models.custom_gru_cell import CustomCGRCell
import utils
from data.data_store import DataStore


def train():
    model.zero_grad()
    pokemon, input_tensor, target_tensor = utils.randomTrainingExample(pokemons)
    hidden = torch.zeros(1, hidden_size)
    all_outputs = []
    for i in range(len(pokemon)):
        hidden, output = model(input_tensor[i, :, :], hidden)
        all_outputs.append(output)

    output = torch.cat(all_outputs, dim=0)
    # print(pokemon)
    # print(output)
    # print("\n")
    # print(target_tensor)
    loss = criterion(output, target_tensor.squeeze())

    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":
    input_size = utils.num_letters
    hidden_size = 200
    output_size = utils.num_letters
    learnin_rate = 0.005
    n_epochs = 500000
    print_every = 1000


    # model = CustomCGRCell(input_size, hidden_size, output_size)
    model = torch.load("trained_models/pokemon_generator-500000-epochs.pt")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learnin_rate)


    data_store = DataStore("data/data.txt")
    data_store.read_raw_data()
    pokemons = data_store.raw_data

    total_loss = 0

    for epoch in range(1, 1 + n_epochs):
        loss = train()
        total_loss += loss 

        if epoch % print_every == 0:
            print("Training {}% --> Current Loss = {}".format((epoch / n_epochs) * 100, total_loss/print_every))
            total_loss = 0

    
    print("Training Complete")
    torch.save(model, "trained_models/pokemon_generator-1000000-epochs.pt")
    print("Model saved!")