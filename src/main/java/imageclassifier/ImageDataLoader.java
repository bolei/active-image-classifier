package imageclassifier;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public class ImageDataLoader {

	private static final String MAPPING_FILE = "file_mapping.csv";
	private static final String LABEL_FILE = "labels.csv";
	private static final String TRAINING_FILE = "training_data.csv";

	public HashMap<Integer, RawImageInstance> loadImageData(String folderPath)
			throws IOException {

		File mapFile = new File(folderPath, MAPPING_FILE);
		File labelFile = new File(folderPath, LABEL_FILE);
		File trainFile = new File(folderPath, TRAINING_FILE);

		if (mapFile.exists() == false) {
			throw new FileNotFoundException(MAPPING_FILE + " not found");
		}
		if (labelFile.exists() == false) {
			throw new FileNotFoundException(LABEL_FILE + " not found");
		}
		if (trainFile.exists() == false) {
			throw new FileNotFoundException(TRAINING_FILE + " not found");
		}
		BufferedReader mapIn = new BufferedReader(new FileReader(mapFile));
		BufferedReader labelIn = new BufferedReader(new FileReader(labelFile));
		BufferedReader trainIn = new BufferedReader(new FileReader(trainFile));

		HashMap<Integer, RawImageInstance> data = new HashMap<>();

		try {
			String mapLine, labelLine, trainLine;
			while (((mapLine = mapIn.readLine()) != null)
					&& ((labelLine = labelIn.readLine()) != null)
					&& ((trainLine = trainIn.readLine()) != null)) {
				try {
					int id = (int) Double.parseDouble(mapLine);
					double[] feature = parseInstanceFeature(trainLine);
					if (data.containsKey(id)) {
						RawImageInstance instance = data.get(id);
						instance.addFeature(feature);
					} else {
						int label = (int) Double.parseDouble(labelLine);
						RawImageInstance instance = new RawImageInstance(id,
								label);
						instance.addFeature(feature);
						data.put(id, instance);
					}

				} catch (Exception e) {
					System.err.println("error processing image:" + mapLine);
					e.printStackTrace();
				}
			}
			return data;
		} finally {
			if (mapIn != null) {
				try {
					mapIn.close();
				} catch (Exception e) {
					e.printStackTrace();
				}
				mapIn = null;
			}
			if (labelIn != null) {
				try {
					labelIn.close();
				} catch (Exception e) {
					e.printStackTrace();
				}
				labelIn = null;
			}
			if (trainIn != null) {
				try {
					trainIn.close();
				} catch (Exception e) {
					e.printStackTrace();
				}
				trainIn = null;
			}
		}
	}

	private double[] parseInstanceFeature(String line) throws Exception {
		double[] feature = new double[128];
		String[] numStrArr = line.split(",");
		if (numStrArr.length != 128) {
			throw new Exception("not 128 numbers in training data line");
		}
		for (int i = 0; i < 128; i++) {
			feature[i] = Double.parseDouble(numStrArr[i]);
		}
		return feature;
	}
}
