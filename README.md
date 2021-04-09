1) Dateset:   NNrate/data/

All the datasets used in this work  are given in the file named "dateset".

2) Descriptor:  NNrate/descriptor/  

In the Java Development Kit enviroment,  run the executable main.java and then input the number of carbon atoms in the console. The topological indices are displayed in the console.
An additional input file  named "graph" is given in src/isomer/. The first line  lists the keyword, which must be reserved. 
From the second line, the two numbers in each line denote the sequence number of the bonded carbon atoms. 

3) Neural network:  NNrate/train/ 

The codes for training is named "nn_rate.py". 
The training date set must be formated as the following five numbers in each line, X1, X2, X3,the index of carbon atom of the broken C-H bond and temperature. 
An example of input files is supplied as "testset" and "trainset". The optimized parameters are given in the file "saved_weights.pth".

4) Prediction: NNrate/predict/

Running nn_predict.py with the provided neural network parameters and descriptors of the target reaction, the predicted rate constant from each model  (or scheme) will be supplied.
