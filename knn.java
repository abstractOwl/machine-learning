import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;

/**
 * A k-nearest neighbors (kNN) classifier.
 * @author AbstractOwl
 */
public class knn {
	private int k;
	private double[][] trainData;
	
	public knn(int k) {
		this.k = k;
		trainData = null;
	}
	
	/**
	 * Comparator 
	 */
	private class EuclideanDistanceComparator implements Comparator<double[]> {
		private double[] origin;
		
		public EuclideanDistanceComparator(double[] origin) {
			this.origin = origin;
		}
		
		private double euclidDistance(double[] arg0, double[] arg1) {
			if (arg0.length != arg1.length) {
				throw new IllegalArgumentException("Array length mismatch");
			}
			
			double accum = 0;
			for (int i = 0; i < arg0.length - 1; ++i) { // Last index is classification
				accum += Math.pow(arg0[i] - arg1[i], 2);
			}
			return Math.sqrt(accum);
		}
		
		@Override
		public int compare(double[] arg0, double[] arg1) {
			double d0 = euclidDistance(origin, arg0);
			double d1 = euclidDistance(origin, arg1);
			return (int) Math.signum(d0 - d1);
		}
	}
	
	/**
	 * Parses a dataset.
	 * @param filename Path to the data file
	 * @param training If not training, pad an extra slot
	 * @return A 2D double array constructed from parsed data
	 */
	private static double[][] parse(String filename, boolean training) {
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
			int N = Integer.parseInt(header[1], 10) + 1;
			result = new double[M][N];
			
			for (int i = 0; i < M; ++i) {
				String[] line = reader.readLine().trim().split("\\s+");
				
				if (N != line.length + (training ? 0 : 1)) {
					reader.close();
					throw new IllegalArgumentException("Expected " + N
							+ " parameters, found " + line.length);
				}
				
				for (int j = 0; j < N; ++j) {
					if (!training && j == N - 1) {
						result[i][j] = -1;
					} else {
						result[i][j] = Double.parseDouble(line[j]);
					}
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
	 * Train the k-nearest neighbors classifier. Stores the training dataset
	 * in memory.
	 * 
	 * @param train Path to training dataset
	 */
	public void train(String train) {
		this.trainData = parse(train, true);
	}
	
	/**
	 * Classify points in the testing dataset based on votes of k nearest
	 * neighbors.
	 * 
	 * @param test Path to testing dataset
	 */
	public void test(String test) {
		double[][] testData = parse(test, false);
		StringBuilder sb = new StringBuilder();
		
		for (int i = 0; i < testData.length; ++i) {
			sb.append(i + 1).append('.');
			
			for (int j = 0; j < testData[i].length; ++j) {
				sb.append(' ').append(testData[i][j]);
			}
			
			PriorityQueue<double[]> heap = new PriorityQueue<double[]>(
					trainData.length,
					new EuclideanDistanceComparator(testData[i])
			);
			
			for (int j = 0; j < trainData.length; ++j) {
				heap.add(trainData[j]);
			}
			
			Hashtable<Integer, Integer> count = new Hashtable<Integer, Integer>();
			double[][] closest = new double[k][testData[i].length];
			for (int j = 0; j < k; ++j) {
				closest[j] = (double[]) heap.remove();
				int index = (int) closest[j][closest[j].length - 1];
				if (count.containsKey(index)) {
					count.put(index, count.get(index) + 1);
				} else {
					count.put(index, new Integer(1));
				}
			}
			
			ArrayList<Map.Entry<Integer, Integer>> list =
					new ArrayList<Entry<Integer, Integer>>(count.entrySet());
			Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {
				@Override
				public int compare(Entry<Integer, Integer> arg0,
						Entry<Integer, Integer> arg1) {
					return (int) Math.signum(arg0.getValue() - arg1.getValue());
				}
			});
			
			// Do k closest neighbors vote
			int highestIndex = -1;
			int highestCount = -1;
			for (int j = 0; j < list.size(); ++j) {
				if (list.get(j).getValue() > highestCount) {
					highestIndex = list.get(j).getKey();
					highestCount = list.get(j).getValue();
				} else if (list.get(j).getValue() == highestCount) { // Tie breaker
					for (k = 0; k < closest.length; k++) {
						int group = (int) closest[k][closest[k].length - 1];
						if (group == list.get(j).getKey()
								|| group == highestIndex) {
							highestIndex = group;
							break;
						}
					}
				}
			}
			sb.append(" -- ").append(highestIndex).append('\n');
		}
		
		System.out.println(sb.toString());
	}
	
	private static void usage() {
		throw new IllegalArgumentException("usage: java knn k train test");
	}
	
	public static void main(String args[]) {
		if (args.length != 3) usage();
		
		knn k = new knn(Integer.parseInt(args[0], 10));
		k.train(args[1]);
		k.test(args[2]);
	}
}
