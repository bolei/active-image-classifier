package imageclassifier;

import java.io.FileInputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class Main {
	public static void main(String[] args) throws Exception {

		if (args.length == 0) {
			System.err.println("usage: xxx");
			return;
		}

		ImageDataLoader dataLoader = new ImageDataLoader();
		DataPreprocessor preProcessor = new DataPreprocessor();

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
			rf[i] = ImageLabelPredictor.createRandomForest(i);
			rf[i].buildClassifier(trainData);
		}

		System.err.println("Classifiers trained");

		String learnerName = prop.getProperty("learner");
		Learner learner = Learner.createLearner(learnerName);
		learner.performClassification(rf, rawInstances, entireData, trainData,
				testData);

	}

}
