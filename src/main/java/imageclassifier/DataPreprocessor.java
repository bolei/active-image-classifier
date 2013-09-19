package imageclassifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class DataPreprocessor {

	public static final int FEATURE_LENTH = 128;

	private static List<Integer> trainImgIds;

	public List<Integer> getTrainImgIds(
			HashMap<Integer, RawImageInstance> rawInstance, int zeroImageSize,
			int oneImageSize) {
		if (trainImgIds == null) {
			// shuffle
			ArrayList<Integer> allIds = new ArrayList<Integer>(
					rawInstance.keySet());
			Collections.shuffle(allIds, new Random(System.currentTimeMillis()));

			// select training data
			List<Integer> zeroImageIds = new LinkedList<Integer>();
			List<Integer> oneImageIds = new LinkedList<Integer>();
			Iterator<Integer> idIt = allIds.iterator();
			int zeroIndex = 0, oneIndex = 0;
			while (idIt.hasNext()
					&& (zeroIndex < zeroImageSize || oneIndex < oneImageSize)) {
				int id = idIt.next();
				if (rawInstance.get(id).getLabel() == 0
						&& zeroIndex < zeroImageSize) {
					zeroImageIds.add(id);
					zeroIndex++;
				}
				if (rawInstance.get(id).getLabel() == 1
						&& oneIndex < oneImageSize) {
					oneImageIds.add(id);
					oneIndex++;
				}
			}
			List<Integer> trainIds = new LinkedList<Integer>(zeroImageIds);
			trainIds.addAll(oneImageIds);
			trainImgIds = trainIds;
		}
		return trainImgIds;
	}

	public Instances getTrainingData(
			HashMap<Integer, RawImageInstance> rawInstances,
			List<Integer> trainIds) {

		// init attributes
		Instances data = createEmptyInstances();

		for (int id : trainIds) {
			RawImageInstance oneImage = rawInstances.get(id);
			String labelStr = Integer.toString(oneImage.getLabel());
			for (double[] feature : oneImage.getFeatures()) {

				Instance inst = new DenseInstance(FEATURE_LENTH + 1);
				for (int i = 0; i < FEATURE_LENTH; i++) {
					inst.setValue(data.attribute(i), feature[i]);
				}
				inst.setValue(data.attribute(FEATURE_LENTH), labelStr);
				data.add(inst);
			}
		}
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public HashMap<Integer, Instances> getDataGroupedByImage(
			HashMap<Integer, RawImageInstance> rawInstances) {
		HashMap<Integer, Instances> entireData = new HashMap<Integer, Instances>();

		for (int id : rawInstances.keySet()) {

			// init attributes
			Instances data = createEmptyInstances();
			RawImageInstance oneInstance = rawInstances.get(id);
			for (double[] feature : oneInstance.getFeatures()) {
				data.add(new DenseInstance(1.0d, feature));
			}
			data.setClassIndex(data.numAttributes() - 1);
			entireData.put(id, data);
		}

		return entireData;
	}

	private Instances createEmptyInstances() {
		ArrayList<Attribute> atts = new ArrayList<Attribute>();
		for (int i = 0; i < FEATURE_LENTH; i++) {
			atts.add(new Attribute("att" + i));
		}
		// Declare the class attribute along with its values
		ArrayList<String> fvClassVal = new ArrayList<String>(2);
		fvClassVal.add("0");
		fvClassVal.add("1");
		atts.add(new Attribute("label", fvClassVal));
		Instances data = new Instances("MyRelation", atts, 0);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
}
