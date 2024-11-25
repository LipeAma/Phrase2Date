import json
import os
from functools import partial
from numpy import array

def int2hotvec(intlist, vocab):
  return [[1 if j == val else 0 for j in range(len(vocab))] for val in intlist]

def phrase2int(string, length, vocab):
    string = string.replace(',', '')
    string = string[:length]
    string = string.lower()
    unk = vocab['<unk>']
    pad = vocab['<pad>']

    int_list = [vocab.get(x, unk) for x in string]
    padded_list = int_list + [pad] * (length - len(string))

    return padded_list

def phrase2hotvec(string, length, vocab):
    return array(int2hotvec(phrase2int(string, length, vocab), vocab))

def date2int(string, vocab):
    return [vocab[x] for x in string]

def date2hotvec(string, vocab):
    return int2hotvec(date2int(string, vocab), vocab)

def hotvec2phrase(hotvec, vocab):
    return ''.join([vocab[str(k)] for k in [n.argmax() for n in hotvec]])

files = ["human_vocab.json", "machine_vocab.json", "dataset.json", "inv_machine_vocab.json"]
for filename in files:
    url = "https://github.com/LipeAma/Phrase2Date/raw/refs/heads/main/json_data/" + filename  # Local filename
    with requests.get(url, stream=True) as r:
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with open(filename, "r") as file:
        globals()[filename[:-5]] = json.load(file)
    os.remove(filename)

formatOut = partial(hotvec2phrase, vocab=inv_machine_vocab)
formatIn = partial(phrase2hotvec, vocab=human_vocab)
