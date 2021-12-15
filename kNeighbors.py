import util
import numpy
import math
import statistics
import tracemalloc
import time

class kNeighborsClassifier:
  def __init__( self, legalLabels, k=10):
    self.type = "kNN"
    self.weights = {}
    self.legalLabels = legalLabels
    self.k = k

  def downscale(self, datum_list):
    #For Digits
    height, width = 28, 28
    img_height, img_width = 4, 4
    img_rows, img_cols = 7, 7

    #For Faces
    if 2 not in self.legalLabels:
      height, width = 70,60
      img_height, img_width = 7,6
      img_rows, img_cols = 10,10

    allscaledDownData = []
    #Iterate through the data in the list
    for data in datum_list:
      scaledDownData = util.Counter()
      #Then iterate through the rows and columns in the image
      for outer in range(img_rows):
        for inner in range(img_cols):
          isFeature = 0
          #Iterate through the image pixels' height and width
          for img_inner in range(img_height):
            if isFeature:
              break
            for img_outer in range(img_width):
              if data[( outer*img_height + img_inner , inner*img_width + img_outer )] == 1:
                isFeature = 1
                break

          scaledDownData[(outer,inner)] = isFeature

      allscaledDownData.append(scaledDownData)

    return allscaledDownData

  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    self.trainingData = self.downscale(trainingData)
    self.trainingLabels = trainingLabels

  def calculateDistance(self, test_datum, train_data):
    distance_diff = test_datum - train_data
    sum = 0
    for value in distance_diff: 
      sum = sum + abs(distance_diff[value])
    return sum
    
  def classify(self, data ):
    """
    Returns most occurring label from the test image's k closest neighbors or 
    choose the training data's shortest distance image's label.
    """
    data = self.downscale(data)

    g = []
    for datum in data:
      distanceValues = []
      # Specify the percentage of training data used to test data
      for i in range(round(len(self.trainingData))):
        distanceValues.append(  (self.calculateDistance(datum,self.trainingData[i]), i)  )
    
      distanceValues.sort()
      distanceValues = distanceValues[:self.k]

      bestK_labels = []
      for distance in distanceValues:
        bestK_labels.append(self.trainingLabels[distance[1]])

      try:
        g.append(statistics.mode(bestK_labels))
      except:
        g.append(bestK_labels[0])


    return g