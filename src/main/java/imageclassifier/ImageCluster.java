package imageclassifier;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import weka.core.Instances;

public class ImageCluster implements Comparable<ImageCluster> {

	private HashMap<Integer, Instances> images = new HashMap<Integer, Instances>();
	private Double clusterLable = null;
	private HashMap<Double, Integer> labelCount;
	private float homogeneousLevel = Float.MAX_VALUE;

	public ImageCluster(HashMap<Integer, Instances> images) {
		this.images = images;
		labelCount = ImageLabelPredictor.getLableCount(images);

		if (labelCount.isEmpty()) {
			this.clusterLable = (double) new Random().nextInt(2);
		} else {
			this.clusterLable = (Double) ImageLabelPredictor
					.getVotedLable(labelCount);
		}

		int maxCount = Integer.MIN_VALUE;
		int totalCount = 0;
		for (double label : labelCount.keySet()) {
			if (labelCount.get(label) > maxCount) {
				maxCount = labelCount.get(label);
			}
			totalCount += labelCount.get(label);
		}
		if (totalCount != 0) {
			homogeneousLevel = ((float) maxCount) / totalCount;
		}

	}

	public void addLable(int pickedImgId, int label) {
		Instances features = images.get(pickedImgId);
		for (int i = 0; i < features.numInstances(); i++) {
			features.instance(i).setClassValue(label);
		}

		labelCount = ImageLabelPredictor.getLableCount(images);

		if (labelCount.isEmpty()) {
			this.clusterLable = (double) new Random().nextInt(2);
		} else {
			this.clusterLable = (Double) ImageLabelPredictor
					.getVotedLable(labelCount);
		}

		int maxCount = Integer.MIN_VALUE;
		int totalCount = 0;
		for (double l : labelCount.keySet()) {
			if (labelCount.get(l) > maxCount) {
				maxCount = labelCount.get(l);
			}
			totalCount += labelCount.get(l);
		}
		if (totalCount != 0) {
			homogeneousLevel = ((float) maxCount) / totalCount;
		}

	}

	public HashMap<Integer, Double> getPredictionResult() {
		HashMap<Integer, Double> result = new HashMap<Integer, Double>();
		for (int imgId : images.keySet()) {
			result.put(imgId, clusterLable);
		}
		return result;
	}

	public boolean isHomogenous() {
		return labelCount.size() == 1;
	}

	public float getHomogenousLevel() {
		return homogeneousLevel;
	}

	public int randomPickUnlabledImageId() {
		ArrayList<Integer> imageIdList = new ArrayList<Integer>();
		for (int imgId : images.keySet()) {
			if (images.get(imgId).instance(0).classIsMissing()) {
				imageIdList.add(imgId);
			}
		}

		if (imageIdList.isEmpty()) {
			return -1;
		}

		int pickedImgId;
		Collections.shuffle(imageIdList);
		pickedImgId = imageIdList.get(0);
		return pickedImgId;
	}

	@Override
	public int compareTo(ImageCluster o) {
		if (this.getHomogenousLevel() > o.getHomogenousLevel()) {
			return 1;
		} else if (this.getHomogenousLevel() > o.getHomogenousLevel()) {
			return -1;
		} else {
			return 0;
		}
	}

	public HashMap<Integer, Instances> getImages() {
		return images;
	}

	public Double getClusterLable() {
		return clusterLable;
	}

	public HashMap<Double, Integer> getLabelCount() {
		return labelCount;
	}

}
