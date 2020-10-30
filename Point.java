/*
 * Craig Glassbrenner
 * 10/25/2020
 * Multiple Linear Regression
 * 
 * This is just a class called 'Point', the point object stores an array of x-values, the total
 * number of values, and the target y-value, used for testing the accuracy of the model.
 */
public class Point {
	double[] values;
	int numVals;
	double targetVal;
	
	public Point(double[] v, int num, double tar) {
		values = v;
		numVals = num;
		targetVal = tar;
	}
}
