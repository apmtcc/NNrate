1) Dateset:   NNrate/data/

All the datasets used in this work  are given in the file named "dateset".

2) Neural network:  NNrate/train/ 

The codes for training is named "nn_rate.py". 
The training date set must be formated as the following five numbers in each line, X1, X2, X3,the index of carbon atom of the broken C-H bond and temperature. 
An example of input files is supplied as "testset" and "trainset". The optimized parameters are given in the file "saved_weights.pth".

3) Prediction: NNrate/predict/

Running nn_predict.py with the provided neural network parameters and descriptors of the target reaction, the predicted rate constant from each model  (or scheme) will be supplied.
