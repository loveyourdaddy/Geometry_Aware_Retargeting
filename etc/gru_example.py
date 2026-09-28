import torch 
import torch.nn as nn
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), batch_first=True)

    def forward(self, input_seq, input_lengths, hidden=None, cell=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        # Forward pass through lstm
        outputs_, (hidden,cell) = self.lstm(packed, (hidden,cell)) # output (num_layer, hidden_size)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs_, batch_first=True) # (batch, num_layer, hidden_size)
        # Sum bidirectional GRU outputs
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# 3 dimension
batch_size = 2
time_step = 35
# mid_size = 5
input_size = 10
output_size = 20 # hidden size 

num_layer = 3
linear = nn.Linear(input_size, input_size)
rnn = EncoderRNN(input_size=input_size, hidden_size=output_size, embedding=linear, n_layers=num_layer) # tuple([mid_size, input_size])

input_seq = torch.zeros((batch_size, time_step, input_size)) # mid_size,
input_lengths = [17,2] # batch size ? # sequences in the same batch must share the same timestep.
hidden = torch.zeros((num_layer, batch_size, output_size)) # mid_size, 
cell   = torch.zeros((num_layer, batch_size, output_size)) # mid_size, 

outputs = []
for f in range(time_step): 
    output, hidden = rnn(input_seq, input_lengths, hidden, cell)
    outputs.append(output)
    
outputs = torch.stack(outputs, dim=1)
