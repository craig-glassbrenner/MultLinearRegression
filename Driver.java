/*
 * Craig Glassbrenner
 * 10/25/2020
 * Multiple Linear Regression
 * 
 * This program implements multiple linear regression. The program implements
 * gradient descent for multiple linear regression and uses the k-fold cross validation 
 * to aid in finding the hyper parameters for the model. 
 */
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Driver {

	// Main driver program, reads command line arguments and sets values if they are given
	// Opens the file for reading, reads the file and parses the data
	// Creates a polynomial for each degree as given via the command line arguments 
	// and for each fold, saving 1 fold for the test/validation set and the other folds for training data
	public static void main(String[] args) throws IOException {
		int kFolds = 5;
		int smallest_degree = 1;
		int largest_degree = smallest_degree;
		double learning_rate = 0.005;
		int epoch_limit = 10000;
		int batch_size = 0;
		int verbosity_level = 1;
		String filename = "";
		
		if(args.length == 0) {
			System.out.println("Need to specify a file.");
			System.exit(0);
		} else if(args.length > 16) {
			System.out.println("Too many command line arguments.");
			System.exit(0);
		} else {
			
			for(int i=0; i < args.length; i++) {
				if(args[i].equals("-f")) {
					i++;
					filename = args[i];
				} else if(args[i].equals("-k")) {
					i++;
					kFolds = Integer.parseInt(args[i]);
				} else if(args[i].equals("-d")) {
					i++;
					smallest_degree = Integer.parseInt(args[i]);
					largest_degree = smallest_degree;
				} else if(args[i].equals("-D")) {
					i++;
					largest_degree = Integer.parseInt(args[i]);
				} else if(args[i].equals("-a")) {
					i++;
					learning_rate = Double.parseDouble(args[i]);
				} else if(args[i].equals("-e")) {
					i++;
					epoch_limit = Integer.parseInt(args[i]);
				} else if(args[i].equals("-m")) {
					i++;
					batch_size = Integer.parseInt(args[i]);
				} else if(args[i].equals("-v")) {
					i++;
					verbosity_level = Integer.parseInt(args[i]);
				} else {
					System.out.println("This is not a valid command line argument.");
					System.exit(0);
				}
			}
			
			if(filename.equals("")) {
				System.out.println("Must specify a file name.");
				System.exit(0);
			}
		}
		
		File file = new File(filename);
		BufferedReader br = new BufferedReader(new FileReader(file));
		List<Point> data = new ArrayList<>();
		
		String line;
		while((line = br.readLine()) != null) {
			String[] vals = line.split(" ");
			double[] v = new double[vals.length - 1];
			double tarVal = 0;
			
			for(int i=0; i < vals.length; i++) {
				if(i == vals.length-1) {
					tarVal = Double.parseDouble(vals[i]);
				} else {
					v[i] = Double.parseDouble(vals[i]);
				}
			}
			
			Point p = new Point(v, v.length, tarVal);
			data.add(p);
		}
		
		br.close();
		LinearRegression lr = new LinearRegression(kFolds, learning_rate, epoch_limit, batch_size, verbosity_level);
		
		System.out.printf("Using %d-Fold cross-validation.\n", kFolds);
		System.out.println("----------------------------------");
		if(verbosity_level == 1) {
			System.out.println("Degree\tTrainMSE\tValidMSE");
		}
		
		for(int i=smallest_degree; i <= largest_degree; i++) {
			
			if(verbosity_level == 2) {
				System.out.printf("* Testing Degree %d\n", i);
				System.out.println("\tTrainMSE\tValidMSE");
			} else if(verbosity_level == 3) {
				System.out.printf("* Testing Degree %d\n", i);
			}
			
			double trainMSE = 0;
			double validMSE = 0;
			for(int j=1; j <= kFolds; j++) {
				double[] error = lr.makePoly(i, j, data);
				trainMSE = trainMSE + error[0];
				validMSE = validMSE + error[1];
				
			}

			trainMSE = trainMSE / (double) kFolds;
			validMSE = validMSE / (double) kFolds;
			if(verbosity_level == 1) {
				System.out.printf("%d\t%.6f\t%.6f\n", i, trainMSE, validMSE);
			} else if(verbosity_level == 3) {
				System.out.println("\t* Averaging across the folds...");
				System.out.printf("\t\tAvgFoldTrainErr: \t%.6f\n", trainMSE);
				System.out.printf("\t\tAvgFoldValidErr: \t%.6f\n", validMSE);
				System.out.println();
			}
		}
	}

}
