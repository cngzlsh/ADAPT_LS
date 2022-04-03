import numpy as np

from CWIs.complex_labeller import Complexity_labeller
from plainifier.plainify import *

class ComplexSentence:
    # Sentence class
    def __init__(self, sentence, label_model, tokeniser, verbose=True, beam_width=3):
        self.sentence = sentence
        self.tokenised_sentence = tokeniser.tokenize(self.sentence)
        self.label_model = label_model
        self.verbose = verbose
        self.beam_width = beam_width

        if self.verbose:
            print(f'Untokenised sentence: {self.sentence}')
            print(f'Tokenised sentence: {self.tokenised_sentence}')

        self.label_complex_words()

    def label_complex_words(self, init=True):
        # applying complexity labeller to the sentence

        Complexity_labeller.convert_format_string(self.label_model, self.sentence)
        if init:
            self.bin_labels = Complexity_labeller.get_bin_labels(self.label_model)[0]
        self.is_complex = True if np.sum(self.bin_labels) >= 1 else False
        self.probs = Complexity_labeller.get_prob_labels(self.label_model)
        
        self.complexity_ranking = np.argsort(np.array(self.bin_labels) * np.array(self.probs))[::-1]
        self.most_complex_word = self.tokenised_sentence[self.complexity_ranking[0]]

        if self.verbose:
            print(f'Complex probs: {self.probs}')
            print(f'Binary complexity labels: {self.bin_labels}')
            
            if self.is_complex:
                print(f'\t Most complex word: {self.most_complex_word} \n')
        
        if not self.is_complex:
            print(f'\t Simplificaiton complete or no complex expression found.\n')
    
    def find_MWEs_w_most_complex_word(self, n_gram, filepath):
        # finds the n-gram mwe of the most complex word in the sentence, if any
        # returns: mwe positions or complex word positions
        
        complex_word_pos = self.complexity_ranking[0]

        if complex_word_pos - n_gram + 1 > 0:
            sliding_start = complex_word_pos - n_gram + 1
        else:
            sliding_start = 0
        
        if complex_word_pos + n_gram - 1 < len(self.complexity_ranking):
            sliding_end = complex_word_pos
        else:
            sliding_end = len(self.complexity_ranking) - n_gram

        with open(filepath, 'r') as f:
            mwes = set(f.read().split('\n')) # make set
            avg_mwe_complexity = 0
            for pos in range(sliding_start, sliding_end + 1):
                possible_mwe = ' '.join(self.tokenised_sentence[pos: pos + n_gram])
                
                if possible_mwe in mwes:
                    
                    if np.mean(self.probs[pos:pos+n_gram]) > avg_mwe_complexity:
                        avg_mwe_complexity = np.mean(self.probs[pos:pos+n_gram])
                        valid_mwes_idx = np.arange(pos, pos+n_gram, 1)
                        mwe_found = possible_mwe
                    else:
                        continue
                        
        if avg_mwe_complexity > 0:
            self.idx_to_plainify = valid_mwes_idx
        else:
            self.idx_to_plainify = [complex_word_pos]
        
    
    def find_all_ngram_mwes(self):
        # returns: self.idx_to_plainify the indices of the longest mwe found
        
        if not self.is_complex:
            raise ValueError('Sentence is not complex')
        
        # give priority to longer MWEs
        n_gram_files = {2: two_gram_mwes_list, 3: three_gram_mwes_list, 4:four_gram_mwes_list}
        
        for n in reversed(range(2,5)):
            self.find_MWEs_w_most_complex_word(n, n_gram_files[n])
            
            if len(self.idx_to_plainify) == n: # if such mwe is found
                break
    
    def one_step_plainify(self):
        idx_start = self.idx_to_plainify[0]
        idx_end = self.idx_to_plainify[-1]+1
        print(f'Found complex word or expression: ### {" ".join(self.tokenised_sentence[idx_start:idx_end])} ###. Plainifying...')
        processed_sentence = tokeniseUntokenise(self.sentence, tokenizer)
        forward_result = getTokenReplacement(processed_sentence, idx_start, len(self.idx_to_plainify), 
                                  tokenizer, model, similm, tokenfreq, embeddings, vocabulary2,
                                  verbose=False, backwards=False, maxDepth=3, maxBreadth=16, alpha=(1/9,6/9,2/9))
        backward_result = getTokenReplacement(processed_sentence, idx_start, len(self.idx_to_plainify),
                                  tokenizer, model, similm, tokenfreq, embeddings, vocabulary2, 
                                  verbose=False, backwards=True, maxDepth=3, maxBreadth=16, alpha=(1/9,6/9,2/9))
        words, scores = aggregateResults((forward_result, backward_result))
        print(f'Suggested top 5 subtitutions: {words[:5]}')
        return words[0].split(' ')
        
    
    def sub_in_sentence(self, substitution):
        # plugs a substitution in the sentence, then updates complexity scores
        substitution_len = len(substitution)
        
        idx_start = self.idx_to_plainify[0]
        idx_end = self.idx_to_plainify[-1]+1
        
        self.tokenised_sentence = self.tokenised_sentence[:idx_start] + substitution + self.tokenised_sentence[idx_end:]
        self.sentence = ' '.join(self.tokenised_sentence)
        self.bin_labels = list(self.bin_labels[:idx_start]) + [0] * substitution_len + list(self.bin_labels[idx_end:])
        self.label_complex_words(init=False)
        print(f'\n\t Sentence after substitution: {self.sentence}\n')
        
    def recursive_greedy_plainify(self, max_steps=float('inf')):
        n = 1
        while self.is_complex and n < max_steps:
            self.find_all_ngram_mwes()
            sub = self.one_step_plainify()
            self.sub_in_sentence(sub)
            n += 1
        print(f'Simplification complete.')
    
    def recursive_beam_search_plainfy(self, beam_width):
        pass