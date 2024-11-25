import json
import os
import requests
from functools import partial
from numpy import array

###################### Download dos vocabulários e do dataset ######################
files = ["human_vocab.json", "machine_vocab.json", "dataset.json", "inv_machine_vocab.json", "inv_human_vocab.json"]
for filename in files:
    url = "https://github.com/LipeAma/Phrase2Date/raw/refs/heads/main/json_data/" + filename  # Local filename
    with requests.get(url, stream=True) as r:
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with open(filename, "r") as file:
        globals()[filename[:-5]] = json.load(file)
    os.remove(filename)



###################### Funções principais ######################
def int2hotvec(intlist, vocab):
  return array([[1 if j == val else 0 for j in range(len(vocab))] for val in intlist])

def phrase2int(string, length, vocab=human_vocab):
    string = string.replace(',', '')
    string = string[:length]
    string = string.lower()
    unk = vocab['<unk>']
    pad = vocab['<pad>']
    int_list = [vocab.get(x, unk) for x in string]
    padded_list = int_list + [pad] * (length - len(string))
    return padded_list

def date2int(string, vocab=machine_vocab):
    return [vocab[x] for x in string]

def hotvec2string(hotvec, vocab):
    return [vocab[str(k)] for k in [n.argmax() for n in hotvec]]



###################### Funções Parciais ######################
def phrase2hotvec(string, length):
    return int2hotvec(phrase2int(string, length), human_vocab)


def date2hotvec(string):
    return int2hotvec(date2int(string), machine_vocab)

hotvec2phrase = partial(hotvec2string, vocab=inv_human_vocab)
hotvec2date = partial(hotvec2string, vocab=inv_machine_vocab)
