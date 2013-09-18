package imageclassifier;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

public class Main {
	public static void main(String[] args) throws Exception {

		int numClassifiers = 3;

		ImageDataLoader dataLoader = new ImageDataLoader();
		DataPreprocessor preProcessor = new DataPreprocessor();
		ImageLabelPredictor predictor = new ImageLabelPredictor();
		PredictionEvaluator evaluator = new PredictionEvaluator();

		// load configurations
		String paramFileName = "config.properties";
		Properties prop = new Properties();
		prop.load(Main.class.getClassLoader()
				.getResourceAsStream(paramFileName));

		HashMap<Integer, RawImageInstance> rawInstances = dataLoader
				.loadImageData(prop.getProperty("dataFolder"));

		int labelSize = Integer.parseInt(prop.getProperty("labelSize"));
		List<Integer> trainIds = preProcessor.getTrainImgIds(rawInstances,
				labelSize);
		Instances trainData = preProcessor.getTrainingData(rawInstances,
				trainIds);
		HashMap<Integer, Instances> entireData = preProcessor
				.getDataGroupedByImage(rawInstances);

		HashMap<Integer, Instances> testData = new HashMap<>(entireData);
		for (int id : trainIds) {
			testData.remove(id);
		}

		// train classifier
		RandomForest[] rf = new RandomForest[numClassifiers];
		for (int i = 0; i < numClassifiers; i++) {
			rf[i] = new RandomForest();
			rf[i].buildClassifier(trainData);
		}

		// get starting accuracy
		float[] accuracies = new float[numClassifiers];
		int numLabelRequest = 0, numImgSeen = 0;
		for (int i = 0; i < numClassifiers; i++) {
			HashMap<Integer, Integer> predictionResults = predictor
					.makeBatchPredictions(rf[i], entireData);
			accuracies[i] = evaluator.getPredictionAccuracy(rawInstances,
					predictionResults);
		}
		evaluator
				.printEvaluationResult(numImgSeen, numLabelRequest, accuracies);

		// begin test
		Set<Integer> labelSet = new HashSet<>();
		for (int id : testData.keySet()) {
			labelSet.clear();
			for (int i = 0; i < numClassifiers; i++) {
				int label = predictor
						.makeOnePrediction(rf[i], testData.get(id));
				labelSet.add(label);
			}
			if (labelSet.size() > 1) { // disagree with the label

				// request label
				int label = rawInstances.get(id).getLabel();
				numLabelRequest++;
				for (int i = 0; i < testData.get(id).numInstances(); i++) {
					Instance inst = testData.get(id).get(i);
					inst.setClassValue(label);
					trainData.add(inst);
				}
				// train again
				for (int i = 0; i < numClassifiers; i++) {
					rf[i] = new RandomForest();
					rf[i].buildClassifier(trainData);
				}

				// reevaluate accuracies
				Arrays.fill(accuracies, 0f);
				for (int i = 0; i < numClassifiers; i++) {
					HashMap<Integer, Integer> predictionResults = predictor
							.makeBatchPredictions(rf[i], entireData);
					accuracies[i] = evaluator.getPredictionAccuracy(
							rawInstances, predictionResults);
				}
				evaluator.printEvaluationResult(numImgSeen, numLabelRequest,
						accuracies);

			}
		}

	}

}
