Craig Glassbrenner
Multiple Linear Regression
10/30/2020

README:

This program is my implmentation of the Multple Linear Regression with Gradient Descent Learn Algorithm. The data set that I used to create this model are:

		sample-p1-d1-no-noise.txt
		sample-p1-d1.txt
		sample-p1-d2-no-noise.txt
		sample-p1-d2.txt
		sample-p2-d1-no-noise.txt
		sample-p2-d1.txt
		sample-p2-d2-no-noise.txt
		sample-p2-d2.txt

The goal is to read in one of these data sets and output a model that best predicts the "target value" of each data point (Minimize the error).

How to Compile this program:
This program was created in java so from the command line you will want to:
javac Driver.java
java Driver -f filename

**You must supply a filename otherwise program will not run**
**Other optional flags**

• -f <FILENAME>: (REQUIRED) Reads training data from the file named <FILENAME> (specified as a String)

-k <INTEGER>: Specifies the number of folds for k-fold cross-validation; default is 5

-d <INTEGER>: Specifies the smallest polynomial degree to evaluate; default is 1

-D <INTEGER>: Specifies the largest polynomial degree to evaluate; if not specified, then only evaluate one degree (the degree value specified through the -d flag or its default value)

-a <DOUBLE>: Specifies the learning rate in mini-batch gradient descent; default is 0.005

-e <INTEGER>: Specifies the epoch limit in mini-batch gradient descent; default is 10000

-m <INTEGER>: Specifies the batch size in mini-batch gradient descent; default is 0, which should be interpreted as specifying batch gradient descent

-v <INTEGER>: Specifies a verbosity level, indicating how much output the program should produce; default is 1 (See the Output section for details)