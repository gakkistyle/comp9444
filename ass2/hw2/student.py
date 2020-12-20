#!/usr/bin/env python3
"""
UNSW 20T3 COMP9444 Ass 2
Gruop Member: Qiwen Zheng(z5240149)
Weighted Score on Vlab (CPU mode):

Description:

Data processing:
Stop words from : https://gist.github.com/sebleier/554280 with some adjustment.Delete some stop words from orginally list
since some words such as 'no' 'not' can be related to determining categories and rating classification.
prepossing: use re module to filter out non-English words and other symbols.
postpossing: I tried methods with only considering 50000 MAX frequently words,sometimes I got better results sometimes not.
            Then I decided not to use the method since it will not improve the result significantly and requires extra computation.

Basic Idea:
I use LSTM framework to do this project since LSTM is designed to achieve high performance on NLP.
LSTM is a special RNN that can learn long-term dependence and has performed very well on a variety of issues and is now widely used.
After applying LSTM to process input embedded words,output to fully connected linear layer could be a good idea.

Model description:
After the vectorization the training data(the function textField.vocab.vectors do that),I put them
to to the tnn.utils.rnn.pack_padded_sequence function,which ... .Then I put the embedded data to
the LSTM.The input size of my LSTM is 200,and hidden size is 300.I set the bidirectional parameter
to be False and number of layer to be 2 with dropout rate to be 0.3,because under this condition
I got the highest score.Then after the LSTM model,two parallel Linear model get the second layer of
LSTM output(index 1,this is also the tuning result).The two linear models are both 300 input unit
size,and 128 output unit size,one is for category,the other one is for rating.Then each of them passing
to another layer of linear model,with both 128 hidden unit input,and 2 output size for rating,5 output
size for category.This structure works good for the training data and it can achieve around 85 weighed score.

Loss class:
My loss function class basically uses pytorch CrossEntropyLoss(),which can do both log softmax and cross entroy loss
calculation.Cross Entropy Loss calculation of loss is still a convex optimization problem. When solving with gradient descent, the convex
optimization problem has good convergence characteristics.The idea is simply return the summation of the two loss
(rating loss and category loss).

Other parameters tuning:
The criterion I chose the following parameters is only based on the higher score I got on validation set.
optimizer: I use Adam optimizer with 0.002 learning rate instead of SGD.
trainValSplit: I preserve it to be 0.8.
batchSize : I preserve it to be 32.
epochs :I preserve it to be 10,more of epochs may result in overfitting on test set.
Dimension:I choose Dim to be 200.Larger dimension can be overfitting but dimension of 200 works well on mu model.

"""

import re
import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    new_list = []
    for i in sample:
        i = re.compile("[^a-zA-Z\s\d]").sub(' ', i)
        if (len(i) > 1):
            new_list.append(i)
    return new_list

#MAX_WORDS = 50000
def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    # """
    # batch = [[x if x<MAX_WORDS else 0 for x in example]
    #       for example in batch]
    return batch

stopWords = {"a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are",
             "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
             "can", "d", "did", "do", "does", "doing", "don", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn", "has", 
             "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
             "if", "in", "into", "is",
            "it", "it's", "its", "itself", "just", "ll", "m",
             "ma", "me", "mightn", "more", "most", "my", "myself", "needn", "now", "o", "of",
             "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
             "over", "own", "re", "s", "same", "shan", "she", "she's", "should", "should've", "so", "some",
             "such", "t", "than", "that",
             "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
             "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
             "was", "we", "were", "what", "when", "where",
             "which", "while", "who", "whom", "why", "will", "with",
             "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
             "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's",
             "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's",
             "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's",
             "when's", "where's", "who's", "why's", "would"}
DIM = 200
wordVectors = GloVe(name='6B', dim=DIM)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    use torch.argmax to get the maximum possible classfication based on the index.
    """
    ratingOutput = torch.argmax(ratingOutput,dim = -1)
    categoryOutput = torch.argmax(categoryOutput,dim = -1)
    return ratingOutput.to(device), categoryOutput.to(device)

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    LSTM with two parallel double layer fully connected linear function.
    """

    def __init__(self):
        super(network, self).__init__()
        hid_s = 300
        self.lstm = tnn.LSTM(input_size=DIM, hidden_size = hid_s ,num_layers = 2 ,bidirectional = False,batch_first = True,dropout = 0.3)
        self.linear_f_rating1 = tnn.Linear(hid_s,128)
        self.linear_f_rating2 = tnn.Linear(128,2)
        self.linear_f_cate1 = tnn.Linear(hid_s,128)
        self.linear_f_cate2 = tnn.Linear(128,5)
        self.relu = tnn.ReLU()

    def forward(self, input, length):
        embeded = tnn.utils.rnn.pack_padded_sequence(input,length.cpu(), batch_first=True)
        output, (hidden,cell) = self.lstm(embeded)
        output_rate = self.linear_f_rating2(self.relu(self.linear_f_rating1(hidden[1])))
        output_cate = self.linear_f_cate2(self.relu(self.linear_f_cate1(hidden[1])))
        return  output_rate,output_cate

class loss(tnn.Module):
    """
    CrossEntropy based loss function, the sum of two losses from rating and category is returned as indication of both loss get involved.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        rate_loss = self.loss(ratingOutput,ratingTarget)
        cate_loss = self.loss(categoryOutput,categoryTarget)
        return rate_loss +  cate_loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.002)
