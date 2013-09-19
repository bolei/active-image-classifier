package imageclassifier;

import java.util.LinkedList;
import java.util.List;

public class RawImageInstance {
	private int imageId;
	private int label;
	private List<double[]> features = new LinkedList<double[]>();

	public RawImageInstance(int id, int label) {
		this.imageId = id;
		this.label = label;
	}

	public void addFeature(double[] feature) {
		features.add(feature);
	}

	public int getImageId() {
		return imageId;
	}

	public int getLabel() {
		return label;
	}

	public List<double[]> getFeatures() {
		return features;
	}

}
