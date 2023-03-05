from pos_tagger import POSTagger, load_data
import argparse

def add_space(sent, chars):
    for char in chars:
        sent = f' {char}'.join(sent.split(char))
    return sent

def main(args):
    sentence = add_space(args.i, '.?!,:;\'"()[]-')
    train_data, dev_data = load_data()
    pos_tagger = POSTagger(train_data)
    pos_tags = pos_tagger.viterbi(sentence.split())
    if not(args.use_nltk):
        print([(word,tag) for (word,tag) in zip(sentence.split(), pos_tags)])
    else:
        from nltk import pos_tag
        print(pos_tag(sentence.split()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('--use_nltk', action='store_true')
    args = parser.parse_args()
    main(args)