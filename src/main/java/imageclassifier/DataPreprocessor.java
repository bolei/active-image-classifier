package imageclassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class DataPreprocessor {

	private static final int FEATURE_LENTH = 128;

	private static List<Integer> trainImgIds;

	public List<Integer> getTrainImgIds(
			HashMap<Integer, RawImageInstance> rawInstance, int labelSize) {
		if (trainImgIds == null) {
			// shuffle
			ArrayList<Integer> allIds = new ArrayList<>(rawInstance.keySet());
			Collections.shuffle(allIds, new Random(System.currentTimeMillis()));

			// select training data
			List<Integer> zeroImageIds = new LinkedList<>();
			List<Integer> oneImageIds = new LinkedList<>();
			Iterator<Integer> idIt = allIds.iterator();
			int zeroIndex = 0, oneIndex = 0;
			while (idIt.hasNext()
					&& (zeroIndex < labelSize || oneIndex < labelSize)) {
				int id = idIt.next();
				if (rawInstance.get(id).getLabel() == 0
						&& zeroIndex < labelSize) {
					zeroImageIds.add(id);
					zeroIndex++;
				}
				if (rawInstance.get(id).getLabel() == 1 && oneIndex < labelSize) {
					oneImageIds.add(id);
					oneIndex++;
				}
			}
			List<Integer> trainIds = new LinkedList<>(zeroImageIds);
			trainIds.addAll(oneImageIds);
			trainImgIds = trainIds;
		}
		return trainImgIds;
	}

	public Instances getTrainingData(
			HashMap<Integer, RawImageInstance> rawInstance,
			List<Integer> trainIds) {

		// init attributes
		ArrayList<Attribute> atts = new ArrayList<>();
		for (int i = 0; i < FEATURE_LENTH; i++) {
			atts.add(new Attribute("att" + i));
		}
		atts.add(new Attribute("label"));

		Instances data = new Instances("MyRelation", atts, 0);

		for (int id : trainIds) {
			RawImageInstance oneInstance = rawInstance.get(id);
			for (double[] feature : oneInstance.getFeatures()) {
				double[] vals = Arrays.copyOf(feature, FEATURE_LENTH + 1);
				vals[FEATURE_LENTH] = oneInstance.getLabel();
				data.add(new DenseInstance(1.0d, vals));
			}
		}
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public HashMap<Integer, Instances> getTestData(
			HashMap<Integer, RawImageInstance> rawInstance) {
		// TODO
		return null;

	}
}
