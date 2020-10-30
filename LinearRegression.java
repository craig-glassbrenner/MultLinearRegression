/*
 * Craig Glassbrenner
 * 10/25/2020
 * Multiple Linear Regression
 * 
 * This class has all the main methods that help create and test the model. 
 */
import java.util.List;
import java.util.Random;

public class LinearRegression {
	int kFolds;
	double learning_rate;
	int epoch_limit;
	int batch_size;
	int verbosity_level;
	int degree;
	int origSize;
	
	int weightUpdates;
	int epochs;
	String stopCond = "";
	
	// Constructor method
	public LinearRegression(int k, double l, int e, int b, int v) {
		kFolds = k;
		learning_rate = l;
		epoch_limit = e;
		batch_size = b;
		verbosity_level = v;
	}
	
	// Implements mini-batch gradient descent 
	public double[] miniBatchGradientDescent(Point[] train) {
		// Creates weight array, assigns all to 0 to start
		double[] weights = new double[train[0].values.length];
		for(int i=0; i < weights.length; i++) {
			weights[i] = 0;
		}
		
		// Keeps track of number of epochs and total iterations
		weightUpdates = 0;
		epochs = 0;
		
		// Calculates the starting cost
		double cost = calculateCost(train, weights);
		if(verbosity_level == 4) {
			System.out.printf("\tEpoch\t   %d (iter\t   %d): Cost = %.9f\n", epochs, epochs, cost);
		}
		else if(verbosity_level == 5) {
			System.out.printf("\tEpoch\t   %d (iter\t   %d): Cost = %.9f\tModel: Y = ", epochs, epochs, cost);
			printModel(weights);
		}
		
		// Stopping Conditions: 1. # epochs >= epoch limit 2. Cost < 10^-10 3. Difference in cost < 10^-10
		while(epochs < epoch_limit) {
			Point[][] mini_batches;
			
			// Generates mini-batches if needed
			if(batch_size == 0) {
				mini_batches = new Point[1][train.length];
				for(int i=0; i < mini_batches.length; i++) {
					for(int j=0; j < train.length; j++) {
						mini_batches[i][j] = train[j];
					}
				}
			}
			else {
				mini_batches = generateMiniBatches(train);
			}
			cost = calculateCost(train, weights);
			
			// Calculates new weights
			for(int i=0; i < mini_batches.length; i++) {
				for(int j=0; j < weights.length; j++) {
					weights[j] = calculateWeight(mini_batches[i], weights, j);
				}
				weightUpdates++;
			}
			epochs++;
			
			// calulates newcost to be used in stopping conditions
			double newCost = calculateCost(train, weights);
			if(epochs % 1000 == 0 && verbosity_level == 4) {
				System.out.printf("\tEpoch\t%d (iter\t%d): Cost = %.9f\n", epochs, epochs, newCost);
			} else if(epochs % 1000 == 0 && verbosity_level == 5) {
				System.out.printf("\tEpoch\t   %d (iter\t   %d): Cost = %.9f\tModel: Y = ", epochs, epochs, cost);
				printModel(weights);
			}
			
			// Stopping condition 2
			if(newCost < Math.pow(10, -10)) {
				stopCond = "Cost ~= 0";
				break;
			// Stopping condition 3
			} else if(Math.abs(cost - newCost) < Math.pow(10, -10)) {
				stopCond = "DeltaCost ~= 0";
				break;
			}
			
			cost = newCost;
		}
		
		if(stopCond.equals("")) {
			stopCond = "Epochs = " + epoch_limit;
		}
		
		if(verbosity_level == 4) {
			System.out.printf("\tEpoch\t%d (iter\t%d): Cost = %.9f\n", epochs, epochs, cost);
			System.out.println("\t* Done with fitting!");
		} else if(verbosity_level == 5) {
			System.out.printf("\tEpoch\t   %d (iter\t   %d): Cost = %.9f\tModel: Y = ", epochs, epochs, cost);
			printModel(weights);
			System.out.println("\t* Done with fitting!");
		}
		return weights;
	}
	
	// Calculates the cost of the polynomial or model that we returned
	public double calculateCost(Point[] train, double[] weights) {
		double innerSum;
		double outerSum = 0;
		
		for(int i=0; i < train.length; i++) {
			innerSum = 0;
			for(int j=0; j < train[i].values.length; j++) {
				innerSum = innerSum + (weights[j]*train[i].values[j]);
			}
			
			outerSum = outerSum + Math.pow((train[i].targetVal - innerSum), 2);
		}
		
		double cost = ((double) 1 / train.length) * outerSum;
		return cost;
	}
	
	// Calculates the new weight that we are changing in the model
	public double calculateWeight(Point[] batch, double[] weights, int change_weight) {
		double newWeight;
		double innerSum = 0;
		double outerSum = 0;
		
		for(int i=0; i < batch.length; i++) {
			for(int j=0; j < batch[i].values.length; j++) {
				innerSum = innerSum + (weights[j]*batch[i].values[j]);
			}
			innerSum = batch[i].targetVal - innerSum;
			outerSum = outerSum + ((-2*batch[i].values[change_weight])*innerSum);
			
		}
		
		outerSum = outerSum / (double) batch.length;
		newWeight = weights[change_weight] - (learning_rate*outerSum);
		
		return newWeight;
	}
	
	// Generates the mini-batches if needed in the model
	public Point[][] generateMiniBatches(Point[] train) {
		Random rand = new Random();
		for(int i=0; i < train.length; i++) {
			int randomIndex = rand.nextInt(train.length);
			Point temp = train[randomIndex];
			train[randomIndex] = train[i];
			train[i] = temp;
		}
		
		Point[][] mini_batches;
		if((train.length % batch_size) == 0) {
			mini_batches = new Point[train.length / batch_size][];
		} else {
			mini_batches = new Point[(train.length / batch_size) + 1][];
		}
		int count = 0;
		
		for(int i=0; i < mini_batches.length; i++) {
			Point[] p;
			
			if((train.length % batch_size) == 0) {
				p = new Point[batch_size];
				for(int j=0; j < p.length; j++) {
					p[j] = train[count];
					count++;
				}
			} else {
				if(i == (mini_batches.length-1)) {
					p = new Point[train.length % batch_size];
				} else {
					p = new Point[batch_size];
				}
				
				for(int j=0; j < p.length; j++) {
					p[j] = train[count];
					count++;
				}
			}
			
			mini_batches[i] = p;
		}
		
		return mini_batches;
	}
	
	// Is the main method for this class, calls all the methods to create the polynomial then outputs information
	// and returns the error for each fold
	public double[] makePoly(int d, int notK, List<Point> data) {
		degree = d;
		origSize = data.get(0).values.length;
		
		int arrSize = data.size() / kFolds;
		Point[] training;
		Point[] test;
		if(kFolds == 1) {
			training = new Point[data.size()];
			test = new Point[0];
		} else {
			training = new Point[data.size() - arrSize];
			test = new Point[arrSize];
		}
		
		int trainingCount = 0;
		int testCount = 0;
		for(int i=0; i < data.size(); i++) {
			if((i < (arrSize*notK)) && (i >= (arrSize*notK - arrSize)) && (kFolds != 1)) {
				test[testCount] = data.get(i);
				testCount++;
			} else {
				training[trainingCount] = data.get(i);
				trainingCount++;
			}
		}
		
		training = augment(training, d);
		if(kFolds != 1) {
			test = augment(test, d);
		}
		
		// Time is takes to train and create the model
		long startTime = System.currentTimeMillis();
		
		if(verbosity_level == 4 || verbosity_level == 5 ) {
			System.out.printf("\t* Holding out Fold %d...\n", notK);
			System.out.println("\t *Beginning mini-batch gradient descent");
			System.out.printf("\t(alpha=%f, epochLimit=%d, batchSize=%d)\n", learning_rate, epoch_limit, batch_size);
		} else if(verbosity_level == 3) {
			System.out.printf("\t* Holding out Fold %d...\n", notK);
		}
		
		double[] weights = miniBatchGradientDescent(training);
		int time = (int) (System.currentTimeMillis() - startTime);
		
		double trainMSE = test(training, weights);
		double testMSE;
		if(kFolds == 1) {
			testMSE = test(training, weights);
		} else {
			testMSE = test(test, weights);
		}
		
		if(verbosity_level == 2) {
			System.out.printf("F_%d\t%.6f\t%.6f\n", notK, trainMSE, testMSE);
		} else if(verbosity_level == 3 || verbosity_level == 4 || verbosity_level == 5) {
			System.out.printf("\t\tTraining took %dms, %d epochs, %d iterations (%.4fms / iteration)\n", time, epochs, weightUpdates, ((double) time / weightUpdates));
			System.out.printf("\t\tGD Stop Condition: %s\n", stopCond);
			System.out.printf("\t\tModel: Y= ");
			printModel(weights);
			
			System.out.printf("\t\tCurFoldTrainErr: \t%.6f\n", trainMSE);
			System.out.printf("\t\tCurFoldValidErr: \t%.6f\n", testMSE);
			System.out.println();
		}
		
		// Returns error array so we can find mean error per degree
		double[] errorArray = {trainMSE, testMSE};
		return errorArray;
	}
	
	// Augments the points in the training set
	public Point[] augment(Point[] training, int d) {
		for(int i=0; i < training.length; i++) {
			double[] orig = training[i].values;
			double[] augment = new double[orig.length*d + 1];
			augment[0] = 1;
			
			for(int j=1; j <= orig.length; j++) {
				augment[j] = orig[j-1];
				
				if(d > 1) {
					int count=j;
					int pow = 2;
					while(count <= d && pow <= d) {
						augment[count+orig.length] = Math.pow(orig[j-1], pow);
						count = count+orig.length;
						pow++;
					}
				}
			}
			
			Point augPoint = new Point(augment, augment.length, training[i].targetVal);
			training[i] = augPoint;
		}
		
		return training;
	}
	
	// Tests the Point[] with the weights w and returns the mean error
	public double test(Point[] t, double[] w) {
		double tot_error = 0;
		for(int i=0; i < t.length; i++) {
			double sum = w[0];
			double error = 0;
			for(int j=1; j < t[i].values.length; j++) {
				sum = sum + (w[j]*t[i].values[j]);
			}
			
			error = t[i].targetVal - sum;
			error = error*error;
			
			tot_error = tot_error + error;
		}
		
		return tot_error / (double) t.length;
	}
	
	public void printModel(double[] weights) {
		System.out.printf("%.4f + ", weights[0]);
		for(int i=1; i <= origSize; i++) {
			System.out.printf("%.4fX%d + ", weights[i], i);
			int count = 2;
			
			while(count <= degree) {
				System.out.printf("%.4fX%d^%d + ", weights[i+origSize], i, count);
				count++;
			}
		}
		
		System.out.println();
	}
}




