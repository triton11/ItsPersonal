import glob
import unicodedata
import string
import csv

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker
EOS = n_letters - 1

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
		and c in all_letters
	)

print(unicode_to_ascii("O'Néàl"))

category_lines_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
category_lines = {}
all_categories = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
n_categories = len(all_categories)


# Read a file and split into lines
def read_lines():
	with open('../data/train_data.csv') as csv_file:
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



import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


import random

# Get a random category and random line from that category
def random_training_pair():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    #print("Random category, line: ", category, line)
    return category, line



# One-hot vector for category
def make_category_input(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return Variable(tensor)

# One-hot matrix of first to last letters (not including EOS) for input
def make_chars_input(chars):
    tensor = torch.zeros(len(chars), n_letters)
    for ci in range(len(chars)):
        char = chars[ci]
        tensor[ci][all_letters.find(char)] = 1
    tensor = tensor.view(-1, 1, n_letters)
    return Variable(tensor)

# LongTensor of second letter to end (EOS) for target
def make_target(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    tensor = torch.LongTensor(letter_indexes)
    #print(tensor)
    return Variable(tensor)


# Make category, input, and target tensors from a random category, line pair
def random_training_set():
    category, line = random_training_pair()
    if len(line) > 1000:
    	line = line[0:1000]

    category_input = make_category_input(category)
    line_input = make_chars_input(line)
    line_target = make_target(line)
    #print("Random Train Set: ", category_input, line_input, line_target)
    return category_input, line_input, line_target

def train(category_tensor, input_line_tensor, target_line_tensor):
    hidden = rnn.init_hidden()
    optimizer.zero_grad()
    loss = 0
    #print(input_line_tensor.size()[0])
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        # print("TRAINING", output, hidden, target_line_tensor[i])
        # print(target_line_tensor[i].size())
        loss += criterion(output, target_line_tensor[i].unsqueeze(0))

    loss.backward()
    optimizer.step()
    
    return output, loss.data.item() / input_line_tensor.size()[0]

import time
import math

def time_since(t):
    now = time.time()
    s = now - t
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

n_epochs = 2000
print_every = 100
plot_every = 5
all_losses = []
loss_avg = 0 # Zero every plot_every epochs to keep a running average
learning_rate = 0.005

rnn = RNN(n_letters, 128, n_letters)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

start = time.time()
max_length = 1000

# Generate given a category and starting letter
def generate_one(category, start_char='A', temperature=0.5):
    category_input = make_category_input(category)
    chars_input = make_chars_input(start_char)
    hidden = rnn.init_hidden()

    output_str = start_char
    
    for i in range(max_length):
        output, hidden = rnn(category_input, chars_input[0], hidden)
        
        # Sample as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Stop at EOS, or add to output_str
        #print(top_i)
        if top_i == EOS:
        	output_str += '\n'
        elif top_i >= len(all_letters):
        	print(top_i)
        else:
        	char = all_letters[top_i]
        	output_str += char
        	chars_input = make_chars_input(char)

    return output_str


for epoch in range(1, n_epochs + 1):
    output, loss = train(*random_training_set())
    loss_avg += loss
    
    if epoch % print_every == 0:
        print('%s (%d %d%%) %.4f' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(generate_one("6", "I", 0.8), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0