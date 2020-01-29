import torch 
import torch.nn as nn

class CustomCGRCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(CustomCGRCell, self).__init__()
        self.hidden_size = hidden_size
        self.grucell = nn.GRUCell(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden):
        hidden = self.grucell(input, hidden) # Used as the hidden state for the next token in the sequence
        output = self.softmax(self.h2o(self.dropout(hidden))) # The output at the current timestep
        return hidden, output