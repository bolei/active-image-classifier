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

		Number label = getVotedLable(labelCount);
		return (Double) label;
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

	public static Number getVotedLable(
			HashMap<? extends Number, ? extends Number> labelCount) {
		Number maxKey = 0d;
		Number maxValue = -1;
		for (Number key : labelCount.keySet()) {
			Number val = labelCount.get(key);
			if (val.doubleValue() > maxValue.doubleValue()) {
				maxValue = labelCount.get(key);
				maxKey = key;
			}
		}
		return maxKey;
	}

	public static HashMap<Double, Integer> getLableCount(
			HashMap<Integer, Instances> images) {
		HashMap<Double, Integer> labelCount = new HashMap<Double, Integer>();

		for (int imgId : images.keySet()) {
			if (images.get(imgId).instance(0).classIsMissing() == false) {
				double label = images.get(imgId).instance(0).classValue();
				if (labelCount.containsKey(label)) {
					labelCount.put(label, labelCount.get(label) + 1);
				} else {
					labelCount.put(label, 1);
				}
			}
		}
		return labelCount;
	}
}
