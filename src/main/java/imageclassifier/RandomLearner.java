package imageclassifier;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class RandomLearner extends AbstractLearner {

	private float prob;

	public RandomLearner(AbstractClassifier[] classifiers, float p) {
		super(classifiers);
		prob = p;
	}

	Random rand = new Random(System.currentTimeMillis());

	@Override
	protected boolean shouldRequestLable(Instances curInstances)
			throws Exception {
		int i = rand.nextInt(100);
		if (i < prob * 100) {
			return true;
		} else {
			return false;
		}
	}
}
