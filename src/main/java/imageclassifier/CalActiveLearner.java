package imageclassifier;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class CalActiveLearner extends Learner {

	private ImageLabelPredictor predictor = new ImageLabelPredictor();
	private PredictionEvaluator evaluator = new PredictionEvaluator();

	@Override
	public void performClassification(AbstractClassifier[] classifiers,
			HashMap<Integer, RawImageInstance> rawInstances,
			HashMap<Integer, Instances> entireData, Instances trainData,
			HashMap<Integer, Instances> testData) throws Exception {
		System.err.println("Evalutating accuracy...");
		int numClassifiers = classifiers.length;
		// get starting accuracy
		float[] accuracies = new float[numClassifiers];
		int numLabelRequest = 0, numImgSeen = 0;
		for (int i = 0; i < numClassifiers; i++) {
			HashMap<Integer, Double> predictionResults = predictor
					.makeBatchPredictions(classifiers[i], entireData);
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
				Double label = predictor.makeOnePrediction(classifiers[i],
						curInstances);
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
					classifiers[i] = ImageLabelPredictor.createRandomForest(i);
					classifiers[i].buildClassifier(trainData);
				}

				System.err.println("Reevaluating the classifiers...");
				// reevaluate accuracies
				Arrays.fill(accuracies, 0f);
				for (int i = 0; i < numClassifiers; i++) {
					HashMap<Integer, Double> predictionResults = predictor
							.makeBatchPredictions(classifiers[i], entireData);
					accuracies[i] = evaluator.getPredictionAccuracy(
							rawInstances, predictionResults);
				}
				evaluator.printEvaluationResult(numImgSeen, numLabelRequest,
						accuracies);
			}
		}
	}

}
