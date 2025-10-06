import numpy as np
import cv2 
import os 

def load_minst_dataset(dataset, path):
    
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    
    for label in labels: 
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            
            X.append(image)
            y.append(label)
            
    # why astype('uint8') -> because labels comes with like '0', '1' strings but we wanna convert them to integer 
    return np.array(X), np.array(y).astype('uint8')


def create_data_mnist(path):
    
    X, y = load_minst_dataset('train', path)
    X_test, y_test = load_minst_dataset('test', path)
    
    return X, y, X_test, y_test 





