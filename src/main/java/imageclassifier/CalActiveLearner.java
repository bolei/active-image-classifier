package imageclassifier;

import java.util.HashSet;
import java.util.Set;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class CalActiveLearner extends AbstractLearner {

	public CalActiveLearner(AbstractClassifier[] classifiers) {
		super(classifiers);
	}

	@Override
	protected boolean shouldRequestLable(Instances curInstances)
			throws Exception {
		Set<Double> labelSet = new HashSet<Double>();
		for (int i = 0; i < classifiers.length; i++) {
			Double label = predictor.makeOnePrediction(classifiers[i],
					curInstances);
			labelSet.add(label);
		}
		boolean should = labelSet.size() > 1;
		if (should) {
			System.err.println("Classifiers disagree. " + labelSet
					+ "\nrequest label");
		}

		return should;
	}
}
