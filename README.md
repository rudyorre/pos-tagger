# Parts-of-Speech Tagger using Viterbi Algorithm and Hidden Markov Models
This project is a Python implementation of a Parts-of-Speech (POS) tagger using the Viterbi algorithm and Hidden Markov Models (HMM). Given an input sentence, the tagger outputs the most likely sequence of POS tags for each word in the sentence.

## Dependencies
- Python 3.x
- [nltk](https://www.nltk.org/)

## Usage
The tagger is implemented in tag.py. To use it, simply run:
```bash
$ python3 tag.py -i "Eddie kicks the ball."
[('Eddie', 'NP'), ('kicked', 'VBD'), ('the', 'AT'), ('ball', 'NN'), ('.', '.')]
```

By default, the tagger uses its own HMM model to tag the sentence. If you want to compare the tagger's output to NLTK's POS tagger's output, you can add the `--use_nltk` argument:
```bash
$ python3 tag.py -i "Eddie kicks the ball." --use_nltk
[('Eddie', 'NNP'), ('kicked', 'VBD'), ('the', 'DT'), ('ball', 'NN'), ('.', '.')]
```

A set of basic tests can be run using `pos_tagger.py` and `test.py`, which show that on our development dataset, the NLTK Parts-of-Speech tagger achieves an accuracy of 62% while this implementation achieves 42%.

## License
This project is licensed under the MIT License.