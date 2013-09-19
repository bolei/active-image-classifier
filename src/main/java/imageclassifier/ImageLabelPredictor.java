package imageclassifier;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class ImageLabelPredictor {
	public double makeOnePrediction(AbstractClassifier classifier,
			Instances onetestData) throws Exception {

		// vote for label
		HashMap<Double, Integer> labelCount = new HashMap<Double, Integer>();
		Iterator<Instance> instanceIt = onetestData.iterator();
		while (instanceIt.hasNext()) {
			double clsLabel = classifier.classifyInstance(instanceIt.next());
			if (labelCount.containsKey(clsLabel)) {
				int count = labelCount.get(clsLabel);
				labelCount.put(clsLabel, count + 1);
			} else {
				labelCount.put(clsLabel, 1);
			}
		}

		double label = getVotedLable(labelCount);
		return label;
	}

	public HashMap<Integer, Double> makeBatchPredictions(
			AbstractClassifier classifier, HashMap<Integer, Instances> testData)
			throws Exception {
		HashMap<Integer, Double> prediction = new HashMap<Integer, Double>();

		for (int imgId : testData.keySet()) {
			double label = makeOnePrediction(classifier, testData.get(imgId));
			prediction.put(imgId, label);
		}
		return prediction;
	}

	public static RandomForest createRandomForest(int seed) {
		RandomForest rf = new RandomForest();
		rf.setNumTrees((int) Math.pow(2, seed + 6));
		rf.setNumFeatures(10 * seed + 20);
		rf.setSeed(new Random().nextInt());
		return rf;
	}

	private double getVotedLable(HashMap<Double, Integer> labelCount) {
		double maxKey = 0d;
		int maxValue = Integer.MIN_VALUE;
		for (double key : labelCount.keySet()) {
			if (labelCount.get(key) > maxValue) {
				maxValue = labelCount.get(key);
				maxKey = key;
			}
		}
		return maxKey;
	}
}
