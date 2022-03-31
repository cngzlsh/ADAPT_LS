# whatever it is called
#import simplifyfunc
import os

# for candidates
# precision: the proportion of generated candidates that are in the gold standard
# recall: the proportion of gold-standard substitutions that are included in the generated substitutions
# f1: harmonic mean between precision and recall

# for SG and SR
# Precision: The proportion with which the replacement
# of the original word is either the original word itself or is in the
# gold standard.
# Accuracy: The proportion with which the replacement
# of the original word is not the original word and is in the gold
# standard
eval_data1 = os.path.join('evaluation', 'BenchLS', 'BenchLS.txt')
eval_data2 = os.path.join('evaluation', 'NNSeval', 'NNSeval.txt')
eval_data3 = os.path.join('evaluation', 'lexmturk', 'lexmturk.txt')

with open(eval_data1) as file:
    sentences = file.readlines()
print(sentences)

def FRES(sentence):
    word_list = sentence.split(' ')
    words = len(word_list)
    syllables = 0
    score = 206.835-1.015 * (words) - 84.6*(syllables/words)



