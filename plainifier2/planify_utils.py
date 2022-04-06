import numpy as np
from tqdm import tqdm

# Load embeddings

def readWord2Vec(path):
	result={}
	for line in open(path, encoding='utf-8'):
		parts=line.strip().split()
		word=parts[0]
		row=np.array([float(x) for x in parts[1:]])
		result[word]=row
	return(result)

def getToken2Vec(embeddingsW, tokenizer):
	result={}
	for word in tqdm(embeddingsW):
		tokens=tokenizer.tokenize(word)
		for token in tokens:
			if token in result:
				result[token]=result[token]+embeddingsW[word]
			else:
				result[token]=embeddingsW[word]
	for token in result:
		result[token]=result[token]/np.sqrt(np.sum(result[token]*result[token]))
	vocabulary=tokenizer.get_vocab()
	table=np.zeros((len(vocabulary),len(result['the'])))
	for token in result:
		table[vocabulary[token],]=result[token]
	return(table)


def readUnigramFreq(path):
	max=0
	result={}
	for line in open(path, encoding='utf-8'):
		parts=line.strip().split()
		word=parts[0]
		frequency=int(parts[1])
		if max==0:
			max=frequency
		result[word]=frequency*1.0/max
	return(result)

def getTokenFreq(unifreq, tokenizer):
	result={}
	for word in tqdm(unifreq):
		tokens=tokenizer.tokenize(word)
		for token in tokens:
			if token in result:
				result[token]=max(result[token],unifreq[word])
			else:
				result[token]=unifreq[word]
	vocabulary=tokenizer.get_vocab()
	table=np.zeros(len(vocabulary))
	for token in result:
		table[vocabulary[token]]=result[token]
	return(table)

def ReadInFile (filename):
    
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines