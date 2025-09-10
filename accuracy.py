import numpy as np

class Accuracy():
    def calculate(self, predictions, y):
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)    
        
        return accuracy
    
class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.presicion = None 

    def init(self, y, reinit=False):
        if self.presicion is None or reinit:
            self.presicion = np.std(y) / 250
        
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.presicion