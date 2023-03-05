import json
import random
import argparse
from collections import defaultdict, Counter

def load_data():
    """
    Loading training and dev data.
    """
    train_path = 'data/train.jsonl' # the data paths are hard-coded 
    dev_path  = 'data/dev.jsonl'

    with open(train_path, 'r') as f:
        train_data = [json.loads(l) for l in f.readlines()]
    with open(dev_path, 'r') as f:
        dev_data = [json.loads(l) for l in f.readlines()]
    return train_data, dev_data

class POSTagger():
    def __init__(self, corpus):
        """
        Args:
            corpus: list of sentences comprising the training corpus. Each sentence is a list
                    of (word, POS tag) tuples.
        """
        # Create a Python Counter object of (tag, word)-frequecy key-value pairs
        self.tag_word_cnt = Counter([(tag, word) for sent in corpus for word, tag in sent])
        # Create a tag-only corpus. Adding the bos token for computing the initial probability.
        self.tag_corpus = [["<bos>"]+[word_tag[1] for word_tag in sent] for sent in corpus]
        # Count the unigrams and bigrams for pos tags
        self.tag_unigram_cnt = self._count_ngrams(self.tag_corpus, 1)
        self.tag_bigram_cnt = self._count_ngrams(self.tag_corpus, 2)
        self.all_tags = sorted(list(set(self.tag_unigram_cnt.keys())))

        # Compute the transition and emission probability 
        self.tran_prob = self.compute_tran_prob()
        self.emis_prob = self.compute_emis_prob()

    def _get_ngrams(self, sent, n):
        """
        Given a text sentence and the argument n, we convert it to a list of n-grams.
        Args:
            sent (list of str): input text sentence.
            n (int): the order of n-grams to return (i.e. 1 for unigram, 2 for bigram, etc.).
        Returns:
            ngrams: a list of n-gram (tuples if n != 1, otherwise strings)
        """
        ngrams = []
        for i in range(len(sent) - n + 1):
            ngram = tuple(sent[i:i+n]) if n != 1 else sent[i]
            ngrams.append(ngram)
        return ngrams

    def _count_ngrams(self, corpus, n):
        """
        Given a training corpus, count the frequency of each n-gram.
        Args:
            corpus (list of str): list of sentences comprising the training corpus with <bos> inserted.
            n (int): the order of n-grams to count (i.e. 1 for unigram, 2 for bigram, etc.).
        Returns:
            ngram_freq (Counter): Python Counter object of (ngram (tuple or str), frequency (int)) key-value pairs.
        """
        corpus_ngrams = []
        for sent in corpus:
            sent_ngrams = self._get_ngrams(sent, n)
            corpus_ngrams += sent_ngrams
        ngram_cnt = Counter(corpus_ngrams)
        return ngram_cnt

    def compute_tran_prob(self):
        """
        Compute the transition probability.

        P(tagB | tagA) = P(tagA | tagB) * P(tagB) / P(tagA) = P(tagB, tagA) / P(tagA)

        Returns:
            tran_prob: a dictionary that maps each (tagA, tagB) tuple to its transition probability P(tagB|tagA).
        """
        tran_prob = defaultdict(lambda: 0) # if a tuple is unseen during training, we set its transition praobility to 0
        for tag_bigram in self.tag_bigram_cnt:
            tagA, _ = tag_bigram
            tran_prob[tag_bigram] = self.tag_bigram_cnt[tag_bigram] / self.tag_unigram_cnt[tagA]
        return tran_prob

    def compute_emis_prob(self):
        """
        Compute the emission probability.

        P(word|tag) = P(tag|word)P(word)/P(tag) = P(tag & word) / P(tag)

        Returns:
            emis_prob: a dictionary that maps each (tagA, wordA) tuple to its emission probability P(wordA|tagA).
        """
        emis_prob = defaultdict(lambda: 0) # if a tuple is unseen during training, we set its emission probability to 0
        for tag, word in self.tag_word_cnt:
            emis_prob[tag, word] = self.tag_word_cnt[tag, word] / self.tag_unigram_cnt[tag]
        return emis_prob

    def init_prob(self, tag):
        """
        Compute the initial probability for a given tag.

        P(tag|<bos>) = P(<bos>, tag) / P(<bos>)

        Returns:
            tag_init_prob (float): the initial probability for {tag}
        """
        return self.tag_bigram_cnt['<bos>', tag] / self.tag_unigram_cnt['<bos>']

    def viterbi(self, sent):
        """
        Given the computed initial/transition/emission probability, make predictions for a given
        sentence using the Viterbi algorithm.

        Implementation based off of https://en.wikipedia.org/wiki/Viterbi_algorithm

        Args:
            sent: a list of words (strings)
        Returns:
            pos_tag: a list of corresponding pos tags (strings)

        Example 1:
            Input: ['Eddie', 'shouted', '.']
            Output: ['NP', 'VBD', '.']
        Example 2:
            Input: ['Mike', 'caught', 'the', 'ball', 'just', 'as', 'the', 'catcher', 'slid', 'into', 'the', 'bag', '.']
            Output: ['NP', 'VBD', 'AT', 'NN', 'RB', 'CS', 'AT', 'NN', 'VBD', 'IN', 'AT', 'NN', '.']
        """
        trellis = {}
        pointers = {}

        # Find tag probabilities for first word
        for tag in self.all_tags:
            trellis[tag, 0] = self.init_prob(tag) * self.emis_prob[tag, sent[0]]
        
        # Find tag probabilities for rest of the words
        for step in range(1, len(sent)):
            word = sent[step]
            for tag in self.all_tags:
                max_tag, max_prob = 'NN', trellis['NN', step - 1] * self.tran_prob['NN', tag] * self.emis_prob[tag, word]
                for prev_tag in self.all_tags:
                    prob = trellis[prev_tag, step - 1] * self.tran_prob[prev_tag, tag] * self.emis_prob[tag, word]
                    if prob > max_prob:
                        max_tag = prev_tag
                        max_prob = prob
                trellis[tag, step] = max_prob
                pointers[tag, step] = max_tag

        # Find most probable final tag
        last_step = len(sent) - 1
        max_tag, max_prob = 'NN', 0.0
        for tag, step in trellis:
            if step == last_step and trellis[tag, step] > max_prob:
                max_tag = tag
                max_prob = trellis[tag, step]

        # Backtrack to find most probable tags at each step
        prev_tag = max_tag
        pos_tag = [prev_tag]        
        for step in range(len(sent)-1, 0, -1):
            prev_tag = pointers[prev_tag, step]
            pos_tag.append(prev_tag)
        pos_tag = pos_tag[::-1]

        return pos_tag

    def test_acc(self, corpus, use_nltk=False):
        """
        Given a training corpus, we compute the model prediction accuracy.
        Args:
            corpus: list of sentences comprising with each sentence being a list
                    of (word, POS tag) tuples
            use_nltk: whether to evaluate the nltk model or our model
        Returns:
            acc: model prediction accuracy (float)
        """
        tot = cor = 0
        for data in corpus:
            sent, gold_tags = zip(*data)
            if use_nltk:
                from nltk import pos_tag
                pred_tags = [x[1] for x in pos_tag(sent)]
            else:
                pred_tags = self.viterbi(sent)
            for gold_tag, pred_tag in zip(gold_tags, pred_tags):
                cor += (gold_tag==pred_tag)
                tot += 1
        acc = cor/tot
        return acc
        
if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=bool, default=True,
            help='Will print information that is helpful for debug if set to True. Passing the empty string in the command line to set it to False.')
    parser.add_argument('--use_nltk', type=bool, default=False,
            help='Whether to evaluate the nltk model. Need to install the package if set to True.')
    parser.add_argument('--input', type=str, help='')
    args = parser.parse_args()

    random.seed(42)
    # Load data
    if args.verbose:
        print('Loading data...')
    train_data, dev_data = load_data()
    if args.verbose:
        print(f'Training data sample: {train_data[0]}')
        print(f'Dev data sample: {dev_data[0]}')

    # Model construction
    if args.verbose:
        print('Model construction...')
    pos_tagger = POSTagger(train_data)
    
    # Model evaluation
    if args.verbose:
        print('Model evaluation...')
    dev_acc = pos_tagger.test_acc(dev_data)
    print(f'Accuracy of our model on the dev set: {dev_acc}')
    if args.use_nltk:
        dev_acc = pos_tagger.test_acc(dev_data, use_nltk=True)
        print(f'Accuracy of the NLTK model on the dev set: {dev_acc}')

    # Tags for custom sentence
    custom_sentence = "Eddie caught the ball .".split()
    tags = pos_tagger.viterbi(custom_sentence) # TODO: Get model predicted tags for the custom sentence
    print(custom_sentence, tags)
    if args.input:
        custom_tags = pos_tagger.viterbi(args.input.split())
        print(args.input.split(), custom_tags)