'''
from nltk.tag import pos_tag

def ReadInFile (filename):
    
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines

norm_test_dat = ReadInFile("./test.8turkers.tok.norm")

with open('./test.8turekers.tok.norm_cap.txt', 'w', encoding='utf-8') as f:

    for sentence in norm_test_dat:
        new_sentence = []
        tokenised = sentence.split(' ')
        
        for token in tokenised:
            try: 
                cap_token = token[0].upper() + token[1:] + ' '
                if pos_tag(pos_tag(cap_token.split(' ')))[0][1] == 'NNP':
                    print(f'NNP: {cap_token}')
                    new_sentence.append(cap_token)
                else:
                    new_sentence.append(token)
            except:
                new_sentence.append(token)
        
        f.writelines(' '.join(new_sentence))
        f.writelines('/n')
'''
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
nltk.download('maxent_ne_chunker')
nltk.download('words')
def get_continuous_chunks(text):
     chunked = ne_chunk(pos_tag(word_tokenize(text)))
     continuous_chunk = []
     current_chunk = []
     for i in chunked:
             if type(i) == Tree:
                     current_chunk.append(" ".join([token for token, pos in i.leaves()]))
             if current_chunk:
                     named_entity = " ".join(current_chunk)
                     if named_entity not in continuous_chunk:
                             continuous_chunk.append(named_entity)
                             current_chunk = []
             else:
                     continue
     return continuous_chunk

my_sent = "WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement."
get_continuous_chunks(my_sent)
print("hello")
