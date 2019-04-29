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
all_categories = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# Read a file and split into lines
def read_lines():
	with open('../data/test_data.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		print("YO")
		for row in csv_reader:
			if (line_count != 0):
				text = unicode_to_ascii(row[1])
				category_lines_list[int(row[0])].append(text)
			line_count += 1

		for i in range(0,16):
			category_lines[i] = category_lines_list[i]
			print(len(category_lines[i]))
# Build the category_lines dictionary, a list of lines per category
lines = read_lines()

