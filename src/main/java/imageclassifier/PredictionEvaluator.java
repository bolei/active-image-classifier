package imageclassifier;

import java.util.HashMap;

public class PredictionEvaluator {
	public float getPredictionAccuracy(
			HashMap<Integer, RawImageInstance> rawInstances,
			HashMap<Integer, Integer> predictionResults) {
		int total = predictionResults.size();
		int correct = 0;
		for (int id : predictionResults.keySet()) {
			if (predictionResults.get(id) == rawInstances.get(id).getLabel()) {
				correct++;
			}
		}

		return ((float) correct) / total;

	}

	public void printEvaluationResult(int numImgSeen, int numLabelRequest,
			float[] accuracies) {
		int arrLen = accuracies.length;
		System.out
				.print(String.format("%d, %d, ", numImgSeen, numLabelRequest));
		for (int i = 0; i < arrLen; i++) {
			System.out.print(accuracies[i] + ", ");
		}
		System.out.println();
	}
}
