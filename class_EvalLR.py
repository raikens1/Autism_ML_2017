import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import f1_score, make_scorer, roc_curve, accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from class_SibKFold import SibKFold

"""
RA, 10/27/2017
class_EvalLR.py
V2
=============================================================
Defines a basic EvalLR object:
Given several input parameters, builds a logistic regression
model, runs k-fold cross validation, and outputs the results
=============================================================
"""

# TODO:
# - [X] test
# - [X] test multiclass functionality
# - [ ] take additional parameters such as scorer, c, etc.
# - [ ] generalize to other classifier types (e.g. SVM)
# - [ ] add function to calculate wald statistic for all coefficients
# - [X] BUGFIX: evaluate() sometimes throws a weird formatting error
# - [X] implement ROC printer
# - [ ] add AU-ROC
# - [ ] add precision-recall
# - [ ] add class function(s) for setting train and test data manually

"""
    EvalLR Object:
    Builds and evaluates a logistic regression model
    Class variables:
        - X (pandas df) Input features for all examples
        - y(1D pandas df) 0/1 labels for all examples
        - model_scores (pandas df) stores train and test scores
        - lr (sk linear model)
        - Xtrain, Xtest (np arrays) store train and test data
        - ytrain, ytrain (np arrays) store train and test labels
"""

class EvalLR(object):
    def __init__(self, X, y, reg = 'l1', multi_class = 'ovr', solver = 'liblinear', c = 1, metric = 'roc'):
        # basic class variables
        self.X = X
        self.y = y
        self.multi_class = multi_class
        self.metric = metric
        
        if self.multi_class == 'multinomial' and solver == 'liblinear':
            print "\'liblinear\' cannot solve multiclass regression. Setting solver to \'newton-cg\' default."
            self.solver = 'newton-cg'
        else:
            self.solver = solver
        
        self.lr = linear_model.LogisticRegression(penalty = reg,
                                                  multi_class = self.multi_class, 
                                                  solver = self.solver, 
                                                  C = c)
        
        # test and train data
        self.Xtrain = None
        self.ytrain = None
        self.Xtest = None
        self.ytest = None

    # print the counts of labels in a label dataset
    def summarize_data(self, labels):
        print labels.ix[:,0].value_counts()
        
    def resample(self, X, y, verbose, oversample = True):
        # subset data by label
        y = pd.DataFrame(y)
        neg = X[y.values == 0]
        pos = X[y.values == 1]

        # resample X to 1:1 ratio
        n_subj = X.shape[0]
        
        if oversample:
            if verbose:
                print "Oversampling from negative controls to 1:1 balance"
            desired = pos.shape[0]
            pos_resamp = pos
        else:
            desired = n_subj/2
            pos_resamp = pos.sample(n = desired, replace = True)
            
        neg_resamp = neg.sample(n = desired, replace = True)

        X_resamp = pd.concat([neg_resamp, pos_resamp])

        # create new labels for resampled data
        neg_y = pd.Series(np.zeros(desired))
        neg_y = neg_y.reindex(neg_resamp.index.values.tolist(), fill_value = 0)

        pos_y = pd.Series(np.zeros(desired))
        pos_y = pos_y.reindex(pos_resamp.index.values.tolist(), fill_value = 1)

        y_resamp = pd.concat([neg_y, pos_y])

        return X_resamp, y_resamp

    # trains on self.Xtrain and self.ytrain
    # evaluates on self.Xtest and self.ytest
    # returns train and test error as a tuple
    def evaluate(self, makeROC, verbose):
        # fit to train data
        self.lr.fit(self.Xtrain, self.ytrain)
        
        # need labels arguement if multiclass
        if self.multi_class == 'multinomial':
            average = 'micro'
            labels = range(len(self.ytest.unique()))
        else:
            average = 'binary'
            labels = None
            
        test_probs = self.lr.predict_proba(self.Xtest)[:,1]
        train_probs = self.lr.predict_proba(self.Xtrain)[:,1]
        
        if self.metric == 'roc':
            # calculate and store roc scores
            testscore = roc_auc_score(self.ytest, test_probs)
            trainscore = roc_auc_score(self.ytrain, train_probs)  
        else:
            fpr, tpr, thresholds = roc_curve(self.ytrain, train_probs, pos_label = 1)
            f1s = [f1_score(self.ytrain, (train_probs>t).astype(int), average = 'binary') for t in thresholds]
            f_i = np.argmax(np.asarray(f1s))
            if verbose:
                print "\nOptimum threshold to maximize f1:", thresholds[f_i]
            trainscore = f1_score(self.ytrain, (train_probs>thresholds[f_i]).astype(int), average = 'binary')
            testscore = f1_score(self.ytest, (test_probs>thresholds[f_i]).astype(int), average = 'binary')
        
        if verbose:
            print "Train Score: %f Test Score: %f" % (trainscore, testscore)
            print self.printMetrics()
        
        if makeROC:
            self.showROC()
        
        return testscore, trainscore

    # runs k-fold cross validation by calling self.evaluate()
    # returns a matrix of model scores
    def kfold(self, k, makeROC = False, verbose = True, resample = False):

        # use KFold from sklearn to split data
        kf = SibKFold(n_splits= k, X = self.X, shuffle = True)

        # make model_scores dataframe to fill in
        model_scores = pd.DataFrame(index = range(1, k+1),
                                    columns = ["Train_score", "Test_score"])
        i = 1
        
        coef = np.zeros(self.X.shape[1])

        for train, test in kf.split(self.X):

            # assign test and train sets
            self.Xtrain, self.Xtest = self.X.iloc[train], self.X.iloc[test]
            self.ytrain, self.ytest = self.y.iloc[train], self.y.iloc[test]
            
            # resample if desired:
            if resample:
                self.Xtrain, self.ytrain = self.resample(self.Xtrain, self.ytrain, verbose)
                self.Xtest, self.ytest = self.resample(self.Xtest, self.ytest, verbose)

            # evaluate and store results
            if verbose:
                print "\nRunning cross validation for fold %d:" % i
                print "==========================================="
                
            testscore, trainscore = self.evaluate(makeROC, verbose)
            coef = np.add(coef, self.lr.coef_)

            model_scores.at[i, 'Test_score'] = testscore
            model_scores.at[i, 'Train_score'] = trainscore

            i += 1

        coef = np.divide(coef, k)
        top_index = np.argsort(coef).tolist()[0][-8:]
        largest_effect = [list(self.X.columns.values)[i] for i in top_index]
        
        return model_scores, largest_effect
    
    def printMetrics(self):
        test_probs = self.lr.predict_proba(self.Xtest)
        train_probs = self.lr.predict_proba(self.Xtrain)
        fpr, tpr, thresholds = roc_curve(self.ytrain, train_probs[:,1], pos_label = 1)
        
        # find threshold that maximizes accuracy
        accs = [accuracy_score(self.ytrain, (train_probs[:,1]>t).astype(int)) for t in thresholds]
        a_i = np.argmax(np.asarray(accs))
        print "AU-ROC", roc_auc_score(self.ytest, test_probs[:,1])

        print "\nOptimum threshold to maximize training accuracy:", thresholds[a_i]
        test_probs_a = (test_probs[:,1]>thresholds[a_i]).astype(int)
        print "F1:", f1_score(self.ytest, test_probs_a, average = 'binary')
        print "Accuracy:", accuracy_score(self.ytest, test_probs_a)
        print "Confusion Matrix:\n", confusion_matrix(self.ytest, test_probs_a)
        

        # find threshold that maximizes f1
        f1s = [f1_score(self.ytrain, (train_probs[:,1]>t).astype(int), average = 'binary') for t in thresholds]
        f_i = np.argmax(np.asarray(f1s))

        test_probs_f = (test_probs[:,1]>thresholds[f_i]).astype(int)
        print "\nOptimum threshold to maximize f1:", thresholds[f_i]
        print "F1:", f1_score(self.ytest, test_probs_f, average = 'binary')
        print "Accuracy:", accuracy_score(self.ytest, test_probs_f)
        print "Confusion Matrix:\n", confusion_matrix(self.ytest, test_probs_f)
       
    
    # makes an ROC plot and shows it
    def showROC(self):
        fpr, tpr, thresholds = self.getROCdata()
        plt.figure()
        lw = 2
        
        if self.multi_class == 'ovr':
            plt.plot(fpr, tpr, color = 'darkorange', lw = lw, label = 'ROC curve')
            
        else: # multiclass
            colors = ['aqua', 'darkorange', 'cornflowerblue']
            for j, color in zip(range(len(self.ytest.unique())), colors):
                plt.plot(fpr[j], tpr[j], color=color, lw=lw,
                         label='ROC curve of class %d' % int(j+1))
                
        plt.plot([0,1], [0,1], color = 'navy', lw = lw, linestyle = '--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.legend(loc='lower right')
        plt.show()
        plt.figure(figsize = (10,10))
        
    def getROCdata(self):
        test_probs = self.lr.predict_proba(self.Xtest)
        
        if self.multi_class == 'ovr':
            fpr, tpr, thresholds = roc_curve(self.ytest, test_probs[:,1], pos_label = 1)
                       
        else: # multinomial
            fpr = {}
            tpr = {}
            thresholds = {}
            for j in range(test_probs.shape[1]):
                fpr[j], tpr[j], thresholds[j] = roc_curve(self.ytest, test_probs[:,j], pos_label = j+1)

        return fpr, tpr, thresholds
    
    def setTrain(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        
    def setTest(self, Xtest, Ytest):
        self.Xtest = Xtest
        self.Ytest = Ytest
   
