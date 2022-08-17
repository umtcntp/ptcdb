import numpy as np
import scipy.sparse as sp
import torch
import pickle
import re
import string
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/ptcdb/", dataset="ptcdb"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    f = open("{}{}.absid".format(path, dataset), "rb")
    df = pickle.load(f)
    f.close()
    print(df)
    abstract_data = df["x"]

    coun_vect = CountVectorizer(lowercase=True,
                                stop_words='english',
                                max_df = 0.1,
                                min_df = 0.001,
                                max_features=10000,
                                preprocessor=custom_preprocessor
                                )
    features = coun_vect.fit_transform(abstract_data)

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    #features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    features1 = features
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    c = flat(edges_unordered)
    b = map(idx_map.get, c)
    a = list(b)
    for i in range(0, len(a)):
        if type(a[i]) == type(None):
            a[i] = 0

    edges = np.array(a).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(6000)
    idx_val = range(6000, 9000)
    idx_test = range(9050, 12450)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    #svm_call(features1, idx_features_labels[:, -1])

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def flat(a):
    l = []
    print(type(a[0][0]))
    for i in a:
        for j in i:
            l.append(j)

    f = np.array(l)

    return f

def custom_preprocessor(text):
    '''
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def svm_call(svm_features, labels):
    print(svm_features)

    X_vec = svm_features
    X_vec = X_vec.todense()

    tfidf = TfidfTransformer() # by default applies "l2" normalization
    X_tfidf = tfidf.fit_transform(X_vec)
    X_tfidf = X_tfidf.todense()

    number_of_round = 10
    accurancies=[]
    precions = []
    recalls = []
    f1_scores= []

    for round in range(number_of_round):
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, 
                                                            test_size = 0.30)
        # Classifier - Algorithm - SVM
        # fit the training dataset on the classifier
        SVM = svm.SVC(C=1.0, kernel='linear', degree=1, gamma='auto')
        #SVM = svm.SVC(kernel='poly',degree=3, probability=True)
        SVM.fit(X_train,y_train)
        # predict the labels on validation dataset
        predictions_SVM = SVM.predict(X_test)
        # Use accuracy_score function to get the accuracy
        accuracy = accuracy_score(predictions_SVM, y_test)*100
        precision = precision_score(predictions_SVM, y_test, average="macro")*100
        recall = recall_score(predictions_SVM, y_test, average="macro")*100
        f1 = f1_score(predictions_SVM, y_test, average="macro")*100
        accurancies.append(accuracy)
        precions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        print("round:"+str(round))


    print("SVM Accuracy Score -> ",sum(accurancies) / len(accurancies))
    print('Precision: %f',sum(precions) / len(precions))
    print('Recall: %f',sum(recalls) / len(recalls))
    print('F1 score: %f',sum(f1_scores) / len(f1_scores))