package imageclassifier;

import java.util.HashMap;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public abstract class Learner {
	public abstract void performClassification(
			AbstractClassifier[] classifiers,
			HashMap<Integer, RawImageInstance> rawInstances,
			HashMap<Integer, Instances> entireData, Instances trainData,
			HashMap<Integer, Instances> testData) throws Exception;

	public static Learner createLearner(String name) {
		if (name.toLowerCase().equals("cal")) {
			return new CalActiveLearner();
		} else {
			return null;
		}
	}
}
