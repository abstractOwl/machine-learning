import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.la4j.LinearAlgebra;
import org.la4j.matrix.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.Vector;
import org.la4j.vector.dense.BasicVector;


/**
 * Provides regression parameters for a training set of data points, given a
 * training set for multivariate linear regression, leading to the regression
 * estimate.
 * 
 * @author AbstractOwl
 */
public class linreg {
	private Vector w;
	private double T;
	
	public linreg() {
		w = null;
		T = 0.0;
	}
	
	/**
	 * Parses a dataset.
	 * @param filename Path to the data file
	 * @return A 2D double array constructed from parsed data
	 */
	private static double[][] parse(String filename) {
		double[][] result;
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new FileReader(filename));
			
			// Read Header
			String[] header = reader.readLine().trim().split("\\s+");
			if (header.length != 2) {
				reader.close();
				throw new IllegalArgumentException("Expected first line of file "
						+ filename + " to be <M> <N>");
			}
			
			// Parse file
			int M = Integer.parseInt(header[0], 10);
			int N = Integer.parseInt(header[1], 10);
			result = new double[M][N];
			
			for (int i = 0; i < M; ++i) {
				String[] line = reader.readLine().trim().split("\\s+");
				
				if (N != line.length) {
					reader.close();
					throw new IllegalArgumentException("Expected " + N
							+ " parameters, found " + line.length);
				}
				
				for (int j = 0; j < N; ++j) {
					result[i][j] = Double.parseDouble(line[j]);
				}
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		} finally {
			try {
				reader.close();
			} catch (IOException e) { /* I tried... */ }
		}
		
		return result;
	}
	
	/**
	 * Computes the w and T value from a training dataset.
	 * @param train Path to train file
	 */
	public void train(String train) {
		double[][] trainData = parse(train);
		double[][] arrayX = new double[trainData.length][trainData[0].length - 1];
		double[]   arrayY = new double[trainData.length];
		
		for (int i = 0; i < trainData.length; ++i) {
			System.arraycopy(trainData[i], 0, arrayX[i], 0, trainData[0].length - 1);
			arrayY[i] = trainData[i][trainData[0].length - 1];
		}
		
		Matrix X = new Basic2DMatrix(arrayX);
		
		// Normalize X
		double[] centroid = new double[X.columns()];
		for (int i = 0; i < X.columns(); i++) {
			centroid[i] = X.getColumn(i).sum() / X.rows();
		}
		double[][] avg_X = new double[X.rows()][X.columns()];
		for (int i = 0; i < avg_X.length; i++) {
			avg_X[i] = centroid;
		}
		X = X.subtract(new Basic2DMatrix(avg_X));
		
		Vector Y = new BasicVector(arrayY);
		
		// Normalize Y
		double avg_Y = Y.sum() / Y.length();
		Y = Y.subtract(avg_Y);
		
		// w = (x*y) / (x^2) => (X_T * X)^-1 * (X_T * y)
		Matrix X_T = X.transpose();
		Matrix m = (X_T.multiply(X)).withInverter(LinearAlgebra.GAUSS_JORDAN)
						.inverse();
		w = m.multiply(X_T).multiply(Y.toColumnMatrix()).toColumnVector();
		// T = y_avg - w * X_centroid
		T = w.multiply(new BasicVector(centroid).toColumnMatrix())
				.subtract(avg_Y).multiply(-1).get(0);
		
		System.out.println("[w, t]: " + w + " " + T);
	}
	
	/**
	 * Computes the regression value, given w and T training values. 
	 * @param test Path to test dataset
	 */
	public void test(String test) {
		if (w == null) {
			throw new IllegalStateException("ERROR: Please run train first");
		}
		
		StringBuilder sb = new StringBuilder();
		double[][] testData = parse(test);
		
		for (int i = 0; i < testData.length; i++) {
			Vector v = new BasicVector(testData[i]);
			double regress = v.multiply(w.toColumnMatrix()).sum() + T;
			
			sb.append(i + 1).append(". ").append(v).append(" -- ")
				.append(regress).append('\n');
		}
		
		System.out.println(sb.toString());
	}
	
	/**
	 * Print usage information.
	 */
	private static void usage() {
		throw new IllegalArgumentException("java linreg <train> <test>");
	}
	
	public static void main(String args[]) {
		if (args.length != 2) {
			usage();
		}
		
		linreg l = new linreg();
		l.train(args[0]);
		l.test(args[1]);
	}
}
