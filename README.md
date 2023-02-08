This project is based on python 3.7 and tenorflow 2.1
'my_ Model. h5' is a trained model.
'project1.py' is the project code file.

'build_ data_ arr()' function is a method to construct continuous layer feature sequence in the research. 
Among them, the atlas with less than 7 layers will increase three times by layer, thus improving the recognition ability of the model for small sequence.

'dataadd()' function is used to expand the training set and add noise.
'x_ Temp_ Deposit 'and' y_ Temp_ Deposit 'stores the x and y corresponding to the dataset that completes the feature sequence construction.

'attention_ 3d_ block() 'is an attention algorithm, but it is not used in later models.
'model_ attention_ applied_ after_ Lstm() 'is the model structure code.
The last section of code draws the model training process and loss diagram
