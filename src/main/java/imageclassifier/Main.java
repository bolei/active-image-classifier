package imageclassifier;

import java.util.HashMap;
import java.util.List;
import java.util.Properties;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class Main {
	public static void main(String[] args) throws Exception {

		// load configurations
		String paramFileName = "config.properties";
		Properties prop = new Properties();
		prop.load(Main.class.getClassLoader()
				.getResourceAsStream(paramFileName));

		ImageDataLoader dataLoader = new ImageDataLoader();
		HashMap<Integer, RawImageInstance> rawInstance = dataLoader
				.loadImageData(prop.getProperty("dataFolder"));

		int labelSize = Integer.parseInt(prop.getProperty("labelSize"));
		DataPreprocessor preProcessor = new DataPreprocessor();
		List<Integer> trainIds = preProcessor.getTrainImgIds(rawInstance,
				labelSize);
		Instances trainData = preProcessor.getTrainingData(rawInstance,
				trainIds);
		HashMap<Integer, Instances> testData = preProcessor
				.getTestData(rawInstance);

		RandomForest rf = new RandomForest();
		rf.buildClassifier(trainData);

		ImageLabelPredictor predictor = new ImageLabelPredictor();
		HashMap<Integer, Integer> prediction = predictor.getPrediction(rf,
				testData);
		// TODO
		System.out.println(prediction);
	}
}
