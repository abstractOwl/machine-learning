import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.la4j.matrix.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.Vector;
import org.la4j.vector.functor.VectorAccumulator;

/**
 * Radial basis function (Gaussian) kernel perceptron.
 * @author AbstractOwl
 */
public class kerpercep {
	private double sigma;
	private int[] alpha;
	private Matrix X;
	private int posEntries;
	public kerpercep() {
		alpha = null;
		X = null;
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
	 * Computes the L2 distance between 2 vectors.
	 * @param x1 Vector 1
	 * @param x2 Vector 2
	 * @return double distance between vectors 
	 */
	private static double euclidDistance(Vector x1, Vector x2) {
		Vector v = x1.subtract(x2);
		return v.fold(new VectorAccumulator() {
			double accumulator = 0.0;
			@Override
			public double accumulate() {
				return Math.sqrt(accumulator);
			}
			@Override
			public void update(int arg0, double arg1) {
				accumulator += arg1 * arg1;
			}
		});
	}
	
	/**
	 * Computes the Gaussian radial basis function.
	 * @param x1 Vector 1
	 * @param x2 Vector 2
	 * @return Radial basis kernel value
	 */
	private double K(Vector x1, Vector x2) {
		return Math.exp(-Math.pow(euclidDistance(x1, x2), 2) / Math.pow(sigma, 2));
	}
	
	/**
	 * Trains the kernel perceptron based on the training dataset.
	 * @param sigma Sigma value to use
	 * @param posTrain Path to positive dataset
	 * @param negTrain Path to negative dataset
	 */
	public void train(double sigma, String posTrain, String negTrain) {
		this.sigma = sigma;
		
		double[][] posData   = parse(posTrain);
		double[][] negData   = parse(negTrain);
		
		// Cache numbers
		int features     = posData[0].length;
		int posEntries   = this.posEntries = posData.length;
		int negEntries   = negData.length;
		int totalEntries = posEntries + negEntries;
		
		// Combine arrays
		double[][] trainData = new double[totalEntries][features];
		for (int i = 0; i < posEntries; ++i) {
			System.arraycopy(posData[i], 0, trainData[i], 0, features);
		}
		for (int i = 0; i < negEntries; ++i) {
			System.arraycopy(negData[i], 0, trainData[posEntries + i], 0, features);
		}
		
		X = new Basic2DMatrix(trainData);
		
		// Normalize X
		double[] centroid = new double[X.columns()];
		for (int i = 0; i < X.columns(); i++) {
			centroid[i] = X.getColumn(i).sum() / X.rows();
		}
		double[][] avg = new double[X.rows()][X.columns()];
		for (int i = 0; i < avg.length; i++) {
			avg[i] = centroid;
		}
		X = X.subtract(new Basic2DMatrix(avg));
		
		
		posData = null;
		negData = null;
		
		alpha = new int[totalEntries];
		boolean converged = false;
		
		while (!converged) {
			converged = true;
			for (int i = 0; i < totalEntries; ++i) {
				int y_i = i >= posEntries ? -1 : 1;
				double sum = 0.0;
				for (int j = 0; j < totalEntries; ++j) {
					int y_j = j >= posEntries ? -1 : 1;
					sum += alpha[j] * y_j * K(X.getRow(i), X.getRow(j));
				}
				if (y_i * sum <= 0) {
					++alpha[i];
					converged = false;
				}
			}
		}
		
		StringBuilder sb = new StringBuilder();
		sb.append("Alphas:");
		for (int i = 0; i < alpha.length; ++i) {
			sb.append(' ').append(alpha[i]);
		}
		sb.append('\n');
		
		System.out.println(sb.toString());
	}
	
	/**
	 * Tests the kernel perceptron.
	 * @param posTest Path to positive dataset
	 * @param negTest Path to negative dataset
	 */
	public void test(String posTest, String negTest) {
		if (alpha == null || X == null) {
			throw new IllegalStateException("ERROR: Please run train before running test.");
		}

		int falseNeg = 0;
		int falsePos = 0;
		
		double[][] negData = parse(negTest);
		double[][] posData = parse(negTest);
		
		// Calculate negatives (+ false positives)
		Matrix Y = new Basic2DMatrix(negData);
		for (int i = 0; i < Y.rows(); ++i) {
			int y_i = -1;
			double sum = 0.0;
			for (int j = 0; j < X.rows(); ++j) {
				int y_j = j < posEntries ? 1 : -1;
				sum += alpha[j] * y_j * K(Y.getRow(i), X.getRow(j));
			}
			if (y_i * sum >= 0) {
				falsePos++;
			}
		}
		System.out.println("False positives: " + falsePos);
		
		// Calculate positives (+ false negatives)
		Y = new Basic2DMatrix(posData);
		for (int i = 0; i < posData.length; ++i) {
			int y_i = 1;
			double sum = 0.0;
			for (int j = 0; j < posData.length; ++j) {
				int y_j = j < posEntries ? 1 : -1;
				sum += alpha[j] * y_j * K(Y.getRow(i), X.getRow(j));
			}
			if (y_i * sum < 0) {
				falseNeg++;
			}
		}
		System.out.println("False negatives: " + falseNeg);
		
		int errorRate = (falseNeg + falsePos) * 100
				/ (negData.length + posData.length);
		System.out.println("Error rate: " + errorRate + "%");
	}
	
	/**
	 * Prints out usage information.
	 */
	private static void usage() {
		throw new IllegalArgumentException(
			"usage: java kerpercep sigma pos_train neg_train pos_test neg_test"
		);
	}
	
	public static void main(String args[]) {
		if (args.length != 5) usage();
		
		kerpercep k = new kerpercep();
		k.train(Double.parseDouble(args[0]), args[1], args[2]);
		k.test(args[3], args[4]);
	}
}
