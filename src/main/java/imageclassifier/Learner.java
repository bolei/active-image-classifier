package imageclassifier;

import java.util.HashMap;

import weka.core.Instances;

public interface Learner {
	public void performClassification(
			HashMap<Integer, RawImageInstance> rawInstances,
			HashMap<Integer, Instances> entireData, Instances trainData,
			HashMap<Integer, Instances> testData) throws Exception;
}
