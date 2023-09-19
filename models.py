# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise. If you do
        spelling correction, this parameter allows you to only use your method for the appropriate dev eval in Q3
        and not otherwise
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.). You will need to implement the predict
    method and you can optionally override predict_all if you want to use batching at inference time (not necessary,
    but may make things faster!)
    """

    def __init__(self, inp, hid, out, word_embeddings):
        self.word_embeddings = word_embeddings
        self.nn = CustomNN(inp, hid, out, word_embeddings)  # FIXME

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        x = np.zeros(self.word_embeddings.get_embedding_length())
        for word in ex_words:
            x += self.word_embeddings.get_embedding(word)
        x /= len(ex_words)
        x = form_input(x)

        log_probs = self.nn.forward(x)
        prediction = torch.argmax(log_probs)

        return prediction


class CustomNN(nn.Module):
    def __init__(self, inp, hid, out, word_embedding):
        super(CustomNN, self).__init__()
        self.word_embedding_layer = word_embedding.get_initialized_embedding_layer()
        self.h = nn.Linear(inp, hid)
        self.output = nn.Linear(hid, out)
        self.non_linear = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=0)  # 1?
        nn.init.xavier_uniform_(self.h.weight)

    def forward(self, x):
        hidden = self.h(x)
        activated = self.non_linear(hidden)
        output = self.output(activated)
        softmaxed = self.log_softmax(output)
        return softmaxed


def form_input(x) -> torch.Tensor:
    """
    Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
    of data, does some computation, handles batching, etc.

    :param x: a [num_samples x inp] numpy array containing input data
    :return: a [num_samples x inp] Tensor
    """
    return torch.from_numpy(x).float()


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings,
                                 train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :param train_model_for_typo_setting: True if we should train the model for the typo setting, False otherwise
    :return: A trained NeuralSentimentClassifier model. Note: you can create an additional subclass of SentimentClassifier
    and return an instance of that for the typo setting if you want; you're allowed to return two different model types
    for the two settings.
    """

    """
    args:
    Namespace(model='DAN', 
    train_path='data/train.txt', 
    dev_path='data/dev.txt', 
    use_typo_setting=False, 
    blind_test_path='data/test-blind.txt', 
    test_output_path='test-blind.output.txt', 
    run_on_test=True, 
    word_vecs_path='data/glove.6B.300d-relativized.txt', 
    lr=0.001, 
    num_epochs=10, 
    hidden_size=100, 
    batch_size=1)
    """

    random.seed(2324)


    len_word_embeddings = word_embeddings.get_embedding_length()
    num_epochs = args.num_epochs
    initial_learning_rate = args.lr # alpha

    ns_classifier = NeuralSentimentClassifier(len_word_embeddings, args.hidden_size, 2,
                                              word_embeddings)
    criterion = nn.CrossEntropyLoss()
    indexer = word_embeddings.word_indexer
    optimizer = optim.Adam(ns_classifier.nn.parameters(), lr=initial_learning_rate)
    for epoch in range(0, num_epochs):

        ex_indices = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_indices)
        total_loss = 0.0

        cnt = 0
        for idx in ex_indices:
            cnt += 1
            if cnt % 1000 == 0:
                print("Epoch",epoch,":", cnt,"/",len(ex_indices))

            x_new = np.zeros(word_embeddings.get_embedding_length())
            for word in train_exs[idx].words:
                x_new += word_embeddings.get_embedding(word)
            x_new /= len(train_exs[idx].words)
            x_new = form_input(x_new)


            #torch.nn.Embedding()
            #    word_embeddings.get_embedding(word)
            # x_onehot = torch.zeros(indexer.__len__())
            # x_onehot.scatter_(0, torch.from_numpy(np.asarray(tmp, dtype=np.int64)), 1)
            # x = form_input(np.asarray(x_onehot))
            y = train_exs[idx].label
            # Build one-hot representation of y. Instead of the label 0 or 1, y_onehot is either [0, 1] or [1, 0]. This
            # way we can take the dot product directly with a probability vector to get class probabilities.
            y_onehot = torch.zeros(2)
            # scatter will write the value of 1 into the position of y_onehot given by y
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.int64)), 1)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ns_classifier.nn.zero_grad()
            log_probs = ns_classifier.nn.forward(x_new)
            # Can also use built-in NLLLoss as a shortcut here but we're being explicit here
            loss = criterion(log_probs, y_onehot)
            total_loss += loss
            # Computes the gradient and takes the optimizer step
            loss.backward()
            optimizer.step()
        print("Total loss on epoch %i: %f" % (epoch, total_loss))
    return ns_classifier
