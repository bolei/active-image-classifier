package imageclassifier;

import java.io.FileInputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class Main {
	public static void main(String[] args) throws Exception {

		if (args.length == 0) {
			System.err.println("usage: xxx");
			return;
		}

		ImageDataLoader dataLoader = new ImageDataLoader();
		DataPreprocessor preProcessor = new DataPreprocessor();
		ImageLabelPredictor predictor = new ImageLabelPredictor();
		PredictionEvaluator evaluator = new PredictionEvaluator();

		// load configurations
		String paramFileName = args[0];
		Properties prop = new Properties();
		prop.load(new FileInputStream(paramFileName));

		System.err.println("Loading raw data...");
		HashMap<Integer, RawImageInstance> rawInstances = dataLoader
				.loadImageData(prop.getProperty("dataFolder"));

		int zeroImageSize = Integer.parseInt(prop.getProperty("zeroImageSize"));
		int oneImageSize = Integer.parseInt(prop.getProperty("oneImageSize"));
		List<Integer> trainIds = preProcessor.getTrainImgIds(rawInstances,
				zeroImageSize, oneImageSize);

		System.err.println("Generating training data...");
		Instances trainData = preProcessor.getTrainingData(rawInstances,
				trainIds);

		System.err.println("Generating entire data...");
		HashMap<Integer, Instances> entireData = preProcessor
				.getDataGroupedByImage(rawInstances);

		System.err.println("Generating test data...");
		HashMap<Integer, Instances> testData = new HashMap<Integer, Instances>(
				entireData);
		for (int id : trainIds) {
			testData.remove(id);
		}

		System.err.println("Training classifiers...");

		int numClassifiers = Integer.parseInt(prop
				.getProperty("numClassifiers"));
		// train classifier
		RandomForest[] rf = new RandomForest[numClassifiers];
		for (int i = 0; i < numClassifiers; i++) {
			rf[i] = predictor.createRandomForest(i);
			rf[i].buildClassifier(trainData);
		}

		System.err.println("Classifiers trained. Tvalutating accuracy...");

		// get starting accuracy
		float[] accuracies = new float[numClassifiers];
		int numLabelRequest = 0, numImgSeen = 0;
		for (int i = 0; i < numClassifiers; i++) {
			HashMap<Integer, Double> predictionResults = predictor
					.makeBatchPredictions(rf[i], entireData);
			accuracies[i] = evaluator.getPredictionAccuracy(rawInstances,
					predictionResults);
		}
		evaluator
				.printEvaluationResult(numImgSeen, numLabelRequest, accuracies);

		System.err.println("Active learning...");

		// begin active learning
		Set<Double> labelSet = new HashSet<Double>();
		for (int id : testData.keySet()) {
			numImgSeen++;
			Instances curInstances = testData.get(id);
			System.err.println("input image: " + id);
			labelSet.clear();
			for (int i = 0; i < numClassifiers; i++) {
				Double label = predictor.makeOnePrediction(rf[i], curInstances);
				labelSet.add(label);
			}
			if (labelSet.size() > 1) { // disagree with the label
				System.err.println("Classifiers disagree. " + labelSet
						+ "\nrequest label");
				// request label
				String labelStr = Integer.toString(rawInstances.get(id)
						.getLabel());
				numLabelRequest++;
				for (int i = 0; i < curInstances.numInstances(); i++) {
					Instance inst = curInstances.get(i);
					double[] oldValues = inst.toDoubleArray();

					Instance newInst = new DenseInstance(
							DataPreprocessor.FEATURE_LENTH + 1);
					for (int j = 0; j < DataPreprocessor.FEATURE_LENTH; j++) {
						newInst.setValue(curInstances.attribute(j),
								oldValues[j]);
					}
					newInst.setValue(curInstances
							.attribute(DataPreprocessor.FEATURE_LENTH),
							labelStr);
					trainData.add(newInst);
				}
				System.err.println("Train again the classifiers...");
				// train again
				for (int i = 0; i < numClassifiers; i++) {
					rf[i] = predictor.createRandomForest(i);
					rf[i].buildClassifier(trainData);
				}

				System.err.println("Reevaluating the classifiers...");
				// reevaluate accuracies
				Arrays.fill(accuracies, 0f);
				for (int i = 0; i < numClassifiers; i++) {
					HashMap<Integer, Double> predictionResults = predictor
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
