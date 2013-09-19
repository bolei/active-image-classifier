package imageclassifier;

import java.util.Arrays;
import java.util.HashMap;

import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public abstract class AbstractStreamingLearner implements Learner {

	protected ImageLabelPredictor predictor = new ImageLabelPredictor();
	protected PredictionEvaluator evaluator = new PredictionEvaluator();
	protected AbstractClassifier[] classifiers;

	public AbstractStreamingLearner(AbstractClassifier[] classifiers) {
		this.classifiers = classifiers;
	}

	@Override
	public void performClassification(
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
		for (int id : testData.keySet()) {
			numImgSeen++;
			System.err.println("input image: " + id);
			Instances curInstances = testData.get(id);

			if (shouldRequestLable(curInstances)) {
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

	protected abstract boolean shouldRequestLable(Instances curInstances)
			throws Exception;

}
