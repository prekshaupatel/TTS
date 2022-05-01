import pickle

f = open("male_test_speakers.pkl", 'rb')
data = pickle.load(f)
ids = [i.split('-')[-1] for i in data]

f = open('test_files.sh', 'w')
f.write(' '.join([i + '.wav' for i in ids]))

f = open('line_index.txt')
data = f.read().splitlines()


test = list()
train = list()

for i in data:
    if i.split()[1] in ids:
        test.append(i)
    else:
        train.append(i)


f = open('line_index_train.txt', 'w')
for i in train:
    f.write(i + '\n')

f = open('line_index_test.txt', 'w')
for i in test:
    f.write(i + '\n')

assert(len(test) == len(ids))
assert(len(data) == len(train) + len(test))
