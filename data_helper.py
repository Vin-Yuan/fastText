import os
import re
import numpy as np
import collections
import ipdb

vocabulary_size = 50000

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def load_data_and_labels(data_dir):
    positive_examples = list(open(os.path.join(data_dir,'rt-polarity.pos'), "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(os.path.join(data_dir,'rt-polarity.neg'), "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # x_text = [x.split(' ') for x in x_text]
    # pos/neg label is 0 or 1
    pos_label = np.zeros(len(positive_examples))
    neg_label = np.ones(len(negative_examples))
    labels = np.concatenate((pos_label, neg_label))
    # Generate labels is [0,1] or [1,0]
    positive_labels = [[0,1] for _ in positive_examples]
    negative_labels = [[1,0] for _ in negative_examples] 
    y = np.concatenate([positive_labels, negative_labels], 0)
    return x_text, y
    return text, label

def build_dataset(text):
    '''
    data: sentence's token_indx, a list of list.
    count: vocabulary count.  'word':cnt.
    dictionary: 'word':index.
    reverse_dictionary: index:'word'
    '''
    count = [['UNK', -1]]
    words = []
    for sentence in text:
        words.extend(sentence.split(' '))
    
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for sentence in text:
        words = sentence.split(' ')
        token_idx = []
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            token_idx.append(index)
        data.append(token_idx)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
def process_data(data, labels, window):
    w_data = []
    label = []

    class_num = len(labels[0])
    count = 0
    for sen_idx, sentence in enumerate(data):
        boundary = max(len(sentence)-window, 0)
        for i in xrange(boundary+1):
            # load current window
            cur_words = sentence[i:i+window]
            pad = window - len(cur_words)
            if pad > 0:
                for x in range(pad):
                    cur_words.append(0)
            w_data.append(cur_words)
            label.append(labels[sen_idx])
            count = count + 1
    return np.array(w_data), np.array(label)

def generate_batch(data, labels, batch_size):
    data_len = len(data)
    iter_step = data_len / batch_size
    last = 0 
    for i in range(iter_step):
        yield data[i:i+batch_size], labels[i:i+batch_size]
        last = i
    yield data[i:], labels[i:]

# may be for half load condition
'''
def generate_batch(data, labels, batch_size, window):
    #data: sentence's token_idx. list of list.
    #return [batch_size, window],[batchi_size, label]
    batch = np.zeros((batch_size, window))
    class_num = len(labels[0])
    label = np.zeros((batch_size, class_num))
    # the load count
    count = 0;
    for sen_idx, sentence in enumerate(data):
        # for current sentence
        boundary = max(len(sentence)-window, 0)
        for i in xrange(boundary+1):
            if count < batch_size:
                #flush the memory writed by last iteration
                batch[count] = np.zeros(window)
                # load current window
                cur_words = sentence[i:i+window]
                pad = window - len(cur_words)
                if pad > 0:
                    for x in range(pad):
                        cur_words.append(0)
                batch[count] = cur_words
                label[count] = labels[sen_idx]
                count = count + 1
            else:
                count = 0;
                yield batch, label
    # may be for half load condition
    yield batch, label
'''        
if __name__ == "__main__":
    text, labels = load_data_and_labels('./data')
    data, count, dictionary, reverse_dictionary = build_dataset(text)
    #batch = generate_batch(data, labels, 64, 6)
    data, labels = process_data(data, labels, 6)
    for x, y in generate_batch(data, labels, 64):
        a =  x.shape
        b = y.shape
        print a,'------',b
