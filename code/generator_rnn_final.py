import glob
import unicodedata
import string
import csv
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import time, math

all_characters = string.printable
n_characters = len(all_characters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_characters
    )

print(unicode_to_ascii("O'Néàl"))

category_lines_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
category_lines = {}
all_categories = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
n_categories = len(all_categories)


# Read a file and split into lines
def read_lines():
    with open('../data/train_data.csv', encoding='ISO-8859-1') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        print("YO")
        for row in csv_reader:
            if (line_count != 0):
                text = unicode_to_ascii(row[1])
                category_lines_list[int(row[0])].append(text)
            line_count += 1

        for i in range(0,16):
            category_lines[str(i)] = category_lines_list[i]
# Build the category_lines dictionary, a list of lines per category
lines = read_lines()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.gru(input.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))

# Turn string into list of longs
def char_tensor(string_in):
    tensor = torch.zeros(len(string_in)).long()
    for c in range(len(string_in)):
        tensor[c] = all_characters.index(string_in[c])
    return Variable(tensor)

def make_target(line):
    letter_indexes = [all_characters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_characters - 1) # EOS
    tensor = torch.LongTensor(letter_indexes)
    return Variable(tensor)

def random_training_set(category):    
    chunk = random.choice(category_lines[category])
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target


def evaluate(start_char='A', temperature=0.8):

    chars_input = char_tensor(start_char)
    hidden = decoder.init_hidden()

    output_str = start_char
    
    for i in range(7000):
        output, hidden = decoder(chars_input[0], hidden)
        
        # Sample as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Stop at EOS, or add to output_str
        char = all_characters[top_i]
        output_str += char
        chars_input = char_tensor(char)

    return output_str

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(inp.size()[0]):
        #print(inp[c])
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))
        #print(loss)

    loss.backward()
    decoder_optimizer.step()

    return loss.data.item() / inp.size()[0]


n_epochs = 2000
print_every = 50
plot_every = 10
hidden_size = 120
n_layers = 2
lr = 0.003

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set("1"))       
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate('I'), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

