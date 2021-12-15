class ClassificationMethod:
  """
  ClassificationMethod is the abstract superclass of 
   - MostFrequentClassifier
   - NaiveBayesClassifier
   - CustomClassifier
 
  As such, you need not add any code to this file.  You can write
  all of your implementation code in the files for the individual
  classification methods listed above.
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    This is the supervised training function for the classifier.  Two sets of 
    labeled data are passed in: a large training set and a small validation set.
    
    Many types of classifiers have a common training structure in practice: using
    training data for the main supervised training loop but tuning certain parameters
    with a small held-out validation set.
    
    To make the classifier generic to multiple problems, the data should be represented
    as lists of Counters containing feature descriptions and their counts.
    """
    abstract
    
  def classify(self, data):
    """
    This function returns a list of labels, each drawn from the set of legal labels
    provided to the classifier upon construction.

    To make the classifier generic to multiple problems, the data should be represented
    as lists of Counters containing feature descriptions and their counts.
    """
    abstract

