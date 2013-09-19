package imageclassifier;

import java.util.Properties;

import weka.classifiers.AbstractClassifier;

public final class LearnerFactory {
	public static Learner createLearner(String name, Properties config,
			AbstractClassifier[] classifiers) {
		if (name.toLowerCase().equals("cal")) {
			return new CalActiveLearner(classifiers);
		} else if (name.toLowerCase().equals("random")) {
			return new RandomLearner(classifiers, Float.parseFloat(config
					.getProperty("p")));
		} else {
			return null;
		}
	}
}
