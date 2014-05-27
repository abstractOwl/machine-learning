import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.la4j.LinearAlgebra;
import org.la4j.matrix.Matrix;
import org.la4j.matrix.dense.Basic2DMatrix;
import org.la4j.vector.Vector;
import org.la4j.vector.dense.BasicVector;

/**
 * Computes the Mahalanobis distances from the centroid of a set of training
 * points to each point in the testing set, using the training data to
 * construct the scatter matrix.
 * 
 * @author AbstractOwl
 */
public class mahadist {
	private Vector centroid;
	private Matrix covariance;
	
	public mahadist() {
		centroid = null;
		covariance = null;
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
	 * Calculates the centroid and covariance matrix of a training data set.
	 * @param train String path of training dataset
	 */
	public void train(String train) {
		double[][] trainData = parse(train);
		double[]   centroid  = new double[trainData[0].length];
		
		// Calculate Centroid
		for (int i = 0; i < trainData[0].length; ++i) {
			for (int j = 0; j < trainData.length; ++j) {
				centroid[i] += trainData[j][i];
			}
			centroid[i] /= trainData.length;
		}
		this.centroid = new BasicVector(centroid);
		
		StringBuilder sb = new StringBuilder();
		sb.append("Centroid: ");
		for (int i = 0, j = centroid.length; i < j; ++i) {
			if (i != 0) sb.append(" ");
			sb.append(centroid[i]);
		}
		System.out.println(sb.toString());
		
		double[][] X_naught = new double[trainData.length][trainData[0].length];
		for (int i = 0, N = X_naught.length; i < N; ++i) {
			for (int j = 0, len = X_naught[0].length; j < len; ++j) {
				X_naught[i][j] = trainData[i][j] - centroid[j];
			}
		}
		Matrix X_n   = new Basic2DMatrix(X_naught);
		Matrix X_n_T = X_n.transpose();
		covariance   = X_n_T.multiply(X_n).divide(trainData.length);
		
		sb = new StringBuilder();
		sb.append("Covariance matrix:\n");
		for (int i = 0; i < covariance.rows(); ++i) {
			if (i != 0) sb.append("\n");
			for (int j = 0; j < covariance.columns(); ++j) {
				if (j != 0) sb.append(" ");
				sb.append(covariance.get(i, j));
			}
		}
		System.out.println(sb.toString());
	}
	
	private static double mahalanobis(Vector x, Vector y, Matrix cov) {
		Vector delta  = x.subtract(y);
		Matrix covInv = cov.withInverter(LinearAlgebra.GAUSS_JORDAN).inverse();
		Matrix tmp    = delta.toRowMatrix().multiply(covInv)
							.multiply(delta.toColumnMatrix());
		return Math.sqrt(tmp.sum());
	}
	
	//private static boolean testMahalanobis() {
	// test cases derived from:
	// http://stat.ethz.ch/education/semesters/ss2012/ams/slides/v2.2.pdf
	//	Vector y = new BasicVector(new double[] { 0, 0 });
	//	Matrix cov = new Basic2DMatrix(new double[][] {
	//			new double[] {25, 0},
	//			new double[] {0, 1}
	//	});
	//	return
	//			(4.0  - mahalanobis(new BasicVector(new double[] { 20, 0 }), y, cov) < 1.0)
	//		&&	(10.0 - mahalanobis(new BasicVector(new double[] { 0, 10 }), y, cov) < 1.0)
	//		&&	(7.3  - mahalanobis(new BasicVector(new double[] { 10, 7 }), y, cov) < 1.0);
	//}
	
	/**
	 * Computes the Mahalanobis distances from several
	 * @param test String path of testing dataset
	 */
	public void test(String test) {
		if (centroid == null || covariance == null) {
			throw new IllegalStateException("Please run train first.");
		}
		
		StringBuilder sb = new StringBuilder();
		sb.append("Distances:\n");
		
		double[][] testingData = parse(test);
		for (int i = 0; i < testingData.length; i++) {
			sb.append(i + 1).append('.');
			for (int j = 0; j < testingData[0].length; j++) {
				sb.append(' ').append(testingData[i][j]);
			}
			sb.append(" -- ");
			Vector v = new BasicVector(testingData[i]);
			sb.append(mahalanobis(v, centroid, covariance)).append('\n');
		}
		System.out.println(sb.toString());
	}
	
	/**
	 * Print usage information.
	 */
	private static void usage() {
		throw new IllegalArgumentException("java mahadist <train> <test>");
	}
	
	public static void main(String args[]) {
		if (args.length != 2) {
			usage();
		}
		
		mahadist m = new mahadist();
		m.train(args[0]);
		m.test(args[1]);
	}
}
