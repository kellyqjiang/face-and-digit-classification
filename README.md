# face-and-digit-classification
Testing Naive Bayes 
(digits)
RUN: python dataClassifier.py  -c naiveBayes -d digits -t 1000

(faces)
RUN: python dataClassifier.py  -c naiveBayes -d faces -t 100

Testing Perceptron
(digits)
RUN: python dataClassifier.py -c perceptron -d digits -t 1000

(faces)
RUN: python dataClassifier.py  -c perceptron -d faces -t 100

Testing kNeighbors
(digits)
RUN: python dataClassifier.py -c kN -d digits -t 1000

(faces)
RUN: python dataClassifier.py  -c kN -d faces -t 100

