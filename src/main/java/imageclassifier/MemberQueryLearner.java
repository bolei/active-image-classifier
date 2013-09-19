package imageclassifier;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeSet;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

public class MemberQueryLearner implements Learner {

	private static final int CLUSTER_NUM = 2;

	private SimpleKMeans clusterer = new SimpleKMeans();

	private PredictionEvaluator evaluator = new PredictionEvaluator();

	public MemberQueryLearner() throws Exception {
		clusterer.setNumClusters(CLUSTER_NUM);
	}

	@Override
	public void performClassification(
			HashMap<Integer, RawImageInstance> rawInstances,
			HashMap<Integer, Instances> entireData, Instances trainData,
			HashMap<Integer, Instances> testData) throws Exception {
		int budget = 31; // 2^5 -1

		ImageCluster initCluster = new ImageCluster(entireData);
		TreeSet<ImageCluster> clusterSet = new TreeSet<ImageCluster>();
		clusterSet.add(initCluster);
		int numLabelRequest = 0;
		do {
			// evaluate result
			doEvaluate(rawInstances, clusterSet, numLabelRequest);

			// do split
			ImageCluster c = pickupHeterogenousCluster(clusterSet);
			if (c == null) {
				break;
			}
			Collections.addAll(clusterSet, split(c));

			int numReq = (budget + 1) / 2;
			for (int i = 0; i < numReq; i++) {
				c = clusterSet.first();
				if (Math.abs(c.getHomogenousLevel() - 1.0) < 0.001) { // Homogeneous
					break;
				}

				// randomly pick points
				int pickedImgId = c.randomPickUnlabledImageId();
				if (pickedImgId < 0) {
					// no points available
					break;
				}
				int label = rawInstances.get(pickedImgId).getLabel();
				budget--;
				c.addLable(pickedImgId, label);
				numLabelRequest++;
				doEvaluate(rawInstances, clusterSet, numLabelRequest);
			}
		} while (budget > 0);
	}

	private void doEvaluate(HashMap<Integer, RawImageInstance> rawInstances,
			TreeSet<ImageCluster> clusterSet, int numLabelRequest) {
		HashMap<Integer, Double> predictionResults = getPredictionResults(clusterSet);
		float accuracy = evaluator.getPredictionAccuracy(rawInstances,
				predictionResults);
		evaluator.printEvaluationResult(0, numLabelRequest,
				new float[] { accuracy });
	}

	private ImageCluster pickupHeterogenousCluster(Set<ImageCluster> clusterSet) {
		Iterator<ImageCluster> it = clusterSet.iterator();
		while (it.hasNext()) {
			ImageCluster c = it.next();
			if (c.isHomogenous() == true) {
				continue;
			}
			it.remove();
			return c;
		}
		return null;
	}

	private Integer clusterImage(Instances imgVals) throws Exception {
		HashMap<Integer, Integer> clusterCount = new HashMap<Integer, Integer>();
		for (int i = 0; i < imgVals.numInstances(); i++) {
			int clust = clusterer.clusterInstance(imgVals.get(i));
			if (clusterCount.containsKey(clust)) {
				clusterCount.put(clust, clusterCount.get(clust) + 1);
			} else {
				clusterCount.put(clust, 1);
			}
		}

		int maxKey = 0;
		int maxValue = Integer.MIN_VALUE;
		for (int key : clusterCount.keySet()) {
			if (clusterCount.get(key) > maxValue) {
				maxValue = clusterCount.get(key);
				maxKey = key;
			}
		}
		return maxKey;
	}

	private HashMap<Integer, Double> getPredictionResults(
			TreeSet<ImageCluster> clusterSet) {
		HashMap<Integer, Double> predictionResults = new HashMap<Integer, Double>();
		for (ImageCluster c : clusterSet) {
			predictionResults.putAll(c.getPredictionResult());
		}
		return predictionResults;
	}

	public ImageCluster[] split(ImageCluster cluster) throws Exception {

		HashMap<Integer, Instances> images = cluster.getImages();

		Instances trainData = null;
		for (int key : images.keySet()) {
			if (trainData == null) {
				trainData = new Instances(images.get(key));
				continue;
			}
			for (int j = 0; j < images.get(key).numInstances(); j++) {
				trainData.add(images.get(key).instance(j));
			}
		}
		if (trainData == null) {
			return null;
		}
		clusterer.buildClusterer(trainData);

		HashMap<Integer, HashMap<Integer, Instances>> splitedImages = new HashMap<Integer, HashMap<Integer, Instances>>();

		for (int key : images.keySet()) {
			int clusterIndex = clusterImage(images.get(key));
			if (!splitedImages.containsKey(clusterIndex)) {
				splitedImages.put(clusterIndex,
						new HashMap<Integer, Instances>());
			}
			splitedImages.get(clusterIndex).put(key, images.get(key));
		}

		ImageCluster[] subClusters = new ImageCluster[CLUSTER_NUM];
		Iterator<Integer> clusterIt = splitedImages.keySet().iterator();

		for (int i = 0; i < CLUSTER_NUM && clusterIt.hasNext(); i++) {
			subClusters[i] = new ImageCluster(splitedImages.get(clusterIt
					.next()));
		}
		return subClusters;
	}
}
