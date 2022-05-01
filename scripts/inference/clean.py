import sys

read_file = sys.argv[1]
write_file = sys.argv[2]

f = open(read_file)
data = f.read().splitlines()

non_numbers = list()
numbers = list()

for i in data:
    num = False
    for ch in i:
        if ch.isdigit():
            numbers.append(i)
            num = True
            break
    if not num:
        non_numbers.append(i)

f = open(write_file, 'w')
for i in non_numbers:
    f.write(i + '\n')

assert(len(data) == len(numbers) + len(non_numbers))
