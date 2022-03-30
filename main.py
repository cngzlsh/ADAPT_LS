import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import copy

from CWIs.complex_labeller import Complexity_labeller
from plainifier.plainify import *

class ComplexSentence:
    # Sentence class
    def __init__(self, sentence, label_model, tokeniser, verbose=True):
        self.sentence = sentence
        self.tokenised_sentence = tokeniser.tokenize(self.sentence)
        self.label_model = label_model
        self.verbose = verbose

        if self.verbose:
            print(f'\t Untokenised sentence: {self.sentence}')
            print(f'\t Tokenised sentence: {self.tokenised_sentence}')

        self.label_complex_words()

    def label_complex_words(self):
        # applying complexity labeller to the sentence

        Complexity_labeller.convert_format_string(self.label_model, self.sentence)
        self.bin_labels = Complexity_labeller.get_bin_labels(self.label_model)
        self.is_complex = True if np.sum(self.bin_labels) >= 1 else False
        self.probs = Complexity_labeller.get_prob_labels(self.label_model)
        
        self.complexity_ranking = np.argsort(self.probs)[::-1]
        self.most_complex_word = self.tokenised_sentence[self.complexity_ranking[0]]

        if self.verbose:
            print(f'\t Complex probs: {self.probs}')
            print(f'\t Binary complexity labels: {self.bin_labels}')
            
            if self.is_complex:
                print(f'\t Most complex word: {self.most_complex_word}')
        
        if not self.is_complex:
            print(f'Simplificaiton complete')
    
    def find_MWEs_w_most_complex_word(self, n_gram, filepath):
        # finds the n-gram mwe of the most complex word in the sentence, if any
        # returns: mwe

        complex_word_pos = self.complexity_ranking[0]
        has_mwe = False

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

            for pos in range(sliding_start, sliding_end + 1):
                possible_mwe = ' '.join(self.tokenised_sentence[pos: pos + n_gram])

                if possible_mwe not in mwes:
                    if self.verbose:
                        print(f'\t Checking MWE "{possible_mwe}": not in list')
                else:
                    valid_mwes_idx = np.arange(pos, pos+n_gram, 1)
                    has_mwe = True
                    if self.verbose:
                        print(f'\t Checking MWE "{possible_mwe}": MWE found!')
        
        if has_mwe:
            return valid_mwes_idx
        else:
            return [complex_word_pos]


def prepare_bert_input(tokeniser, tokenised_sentence, complex_word_idx, num_masks, max_seq_len=128, mask_prob=0.5):
    '''
    Given a sentence, the complex word indices and the number of mask, prepare the sentence to pass into BERT
        returns a dictionary {unused_id, tokens, input_ids, input_mask, input_type_ids} where:
            - unused_ids: 0
            - tokens: list of tokens eg. ['[CLS]', 'a', 'cat', 'perched', '[MASK]', 'a', '[MASK]', '[SEP]', 'a'. 'cat', '[MASK]', 'on', 'a', 'mat', '['CLS]']
            - input_ids: tokens converted by a tokeniser, padded to max_seq_len
            - input_mask: where input_ids are valid
            - input_type_ids: where the original sentence is (incl. CLS token)
            - complex_word_position: list of complex word/mwe positions in input_ids.
            - num_masks: length of substitution words
    '''

    if isinstance(complex_word_idx, np.int64):
        mask_start_pos = complex_word_idx
        mask_end_pos = complex_word_idx
    elif isinstance(complex_word_idx, np.ndarray):
        mask_start_pos = complex_word_idx[0]
        mask_end_pos = complex_word_idx[-1]
    else:
        raise ValueError('Complex word index must be np.int64 or np.ndarray')

    # complex word or MWE
    complex_word = tokenised_sentence[mask_start_pos:mask_end_pos+1]

    # first sentence: random mask except on the complex word or MWE
    tokens = ['[CLS]']
    input_type_ids = [0]

    for tokenised_word in tokenised_sentence:
        if tokenised_word not in complex_word:
            if np.random.random() < mask_prob:
                tokens.append('[MASK]')
            else:
                tokens.append(tokenised_word)
        else:
            tokens.append(tokenised_word)
        input_type_ids.append(0)
    
    # seperator token
    tokens.append('[SEP]')
    input_type_ids.append(0)

    # second sentence: mask the complex word or MWE, not other words
    for tokenised_word in tokenised_sentence[:mask_start_pos]:
        tokens.append(tokenised_word)
        input_type_ids.append(1)

    for _ in range(num_masks):
        tokens.append('[MASK]')
        input_type_ids.append(1)

    for tokenised_word in tokenised_sentence[mask_end_pos+1:]:
        tokens.append(tokenised_word)
        input_type_ids.append(1)
    
    tokens.append('[CLS]')
    input_type_ids.append(1)

    # convert tokens to ids
    input_ids = tokeniser.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # pad the shit
    if len(tokens) > max_seq_len:
        raise ValueError(f'Sentence to be processed has length {len(tokens)} which is more than maximum available {max_seq_len} ')
    else:
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
    
    complex_word_position = np.arange(len(tokenised_sentence) + 2 + mask_start_pos, len(tokenised_sentence) + 2 + mask_end_pos + 1, 1)
    
    assert len(input_type_ids) == max_seq_len
    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len

    bert_input = {'unused_id':0, 'tokens':tokens, 'input_ids':input_ids, 'input_mask':input_mask, 'input_type_ids': input_type_ids, 'complex_word_position':complex_word_position, 'num_masks': num_masks}
    return bert_input

def generate_substitutions_candidates(bert_model, bert_input, tokeniser, topk=80):
    '''
    Given bert_input dictionary (see preprare_bert_input function), generates candidate substitutions using BERT
    returns a list of len(num_masks) of substitution candidates in the position:
        [mask_1_candidates, mask_2_candidates, ...]
        where mask_i_candidates = [bert_output_values (in cpu), bert_output_tokens]
    '''
    # from bert_input
    complex_word_position = bert_input['complex_word_position']
    num_masks = bert_input['num_masks']
    ids_tensor = torch.tensor([bert_input['input_ids']]).to(device)
    type_id_tensor = torch.tensor([bert_input['input_type_ids']]).to(device)
    attention_mask_tensor = torch.tensor([bert_input['input_mask']]).to(device)

    # pass the shit into bert
    with torch.no_grad():
        pred = bert_model(ids_tensor, type_id_tensor, attention_mask_tensor)

    substitution_candidates = []

    substitute_positions = np.arange(complex_word_position[0], complex_word_position[0] + num_masks, 1)
    candidate_values, candidate_ids = pred[0, substitute_positions].topk(topk)
    
    for i in range(num_masks):
        tokens = tokeniser.convert_ids_to_tokens(candidate_ids[i].cpu().numpy())
        substitution_candidates.append([candidate_values[i].cpu().numpy(), tokens])

    return substitution_candidates




if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)

    two_gram_mwes_list = './CWIs/2_gram_mwe_50.txt'
    three_gram_mwes_list = './CWIs/3_gram_mwe_25.txt'
    four_gram_mwes_list = './CWIs/4_gram_mwe_8.txt'
    pretrained_model_path = './CWIs/cwi_seq.model'
    temp_path = './CWIs/temp_file.txt'

    path = './plainifier/'
    premodel = 'bert-large-uncased-whole-word-masking'
    bert_dict = 'tersebert_pytorch_1_0.bin'
    embedding = 'crawl-300d-2M-subword.vec'
    unigram = 'unigrams-df.tsv'
    tokenizer = BertTokenizer.from_pretrained(premodel)
    Complexity_labeller_model = Complexity_labeller(pretrained_model_path, temp_path)

    model, similm, tokenfreq, embeddings, vocabulary2 = load_all(path, premodel, bert_dict, embedding, unigram, tokenizer)
    
 
    input_sentence = 'You ought to have offered to help'
    s = ComplexSentence(input_sentence, label_model=Complexity_labeller_model, tokeniser=tokenizer)
    complex_word_idx = s.find_MWEs_w_most_complex_word(n_gram=2, filepath=two_gram_mwes_list)

    processed_sentence = tokeniseUntokenise(input_sentence, tokenizer)
    result1 = getTokenReplacement(processed_sentence, 
                                complex_word_idx[0], 
                                len(complex_word_idx), 
                                tokenizer,
                                model, 
                                similm, 
                                tokenfreq, 
                                embeddings, 
                                vocabulary2, 
                                verbose=True, 
                                backwards=False, 
                                maxDepth=3, 
                                maxBreadth=16, 
                                alpha=(1/9,6/9,2/9))

    result2 = getTokenReplacement(processed_sentence,
                                complex_word_idx[0],
                                len(complex_word_idx),
                                tokenizer,
                                model, 
                                similm, 
                                tokenfreq, 
                                embeddings, 
                                vocabulary2,
                                verbose=True,
                                backwards=True,
                                maxDepth=3,
                                maxBreadth=16,
                                alpha=(1/9,6/9,2/9))

    words, scores = aggregateResults((result1,result2))