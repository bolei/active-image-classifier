package imageclassifier;

import java.util.HashMap;
import java.util.Iterator;

import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class ImageLabelPredictor {
	public int makeOnePrediction(AbstractClassifier classifier,
			Instances onetestData) throws Exception {

		// vote for label
		HashMap<Double, Integer> labelCount = new HashMap<>();
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

		int label = getVotedLable(labelCount);
		return label;
	}

	public HashMap<Integer, Integer> makeBatchPredictions(
			AbstractClassifier classifier, HashMap<Integer, Instances> testData)
			throws Exception {
		HashMap<Integer, Integer> prediction = new HashMap<>();

		for (int imgId : testData.keySet()) {
			int label = makeOnePrediction(classifier, testData.get(imgId));
			prediction.put(imgId, label);
		}
		return prediction;
	}

	private int getVotedLable(HashMap<Double, Integer> labelCount) {
		double maxKey = 0d;
		int maxValue = Integer.MIN_VALUE;
		for (double key : labelCount.keySet()) {
			if (labelCount.get(key) > maxValue) {
				maxValue = labelCount.get(key);
				maxKey = key;
			}
		}
		return (int) maxKey;
	}
}
