package experiments;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import gep.GEPMTLR;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import utils.WriteReadFromFile;

public class MyExperimentLocal {

	public static void main(String args[]) {

		try {

			WriteReadFromFile file = new WriteReadFromFile();

			ArrayList<String> datasets = file.read("configuration/datasets.cfg");
			//HashMap<String, int[]> hRanges = file.getRange("configuration/h.cfg");

			int numberOfIndividuals = 50;
			int numberOfGenerations = 200;

			for (String dataset : datasets) {

				int kMax = 60;

				double bestRelRMSE = Double.MAX_VALUE;
				double bestDev = 0;
				int bestH = 0;
				int bestK = 0;

				String arr[] = dataset.split("\t");
				String data = arr[0];

				int q = Integer.parseInt(arr[1]);

				//int range[] = hRanges.get(data);

				// load the dataset
				MultiLabelInstances full = new MultiLabelInstances("datasets/modified/" + data + "-modified.arff", q);

				// This case corresponds to Andro dataset
				if (full.getNumInstances() < kMax)
					kMax = 30;

				for (int h = 10; h <= 50; h += 5) {

					String path = "results/" + data + "/GEPMTR-Local/h" + h;

					for (int k = 10; k <= kMax; k += 5) {

						// Constructing the classifier
						GEPMTLR learner = new GEPMTLR(h, numberOfIndividuals, numberOfGenerations, k);

						Evaluator eval = new Evaluator();
						MultipleEvaluation results = null;

						results = eval.crossValidate(learner, full, 10);

						String pT = path + "-k" + k;

						file.createDirectories(path);

						PrintWriter writer = new PrintWriter(new File(pT));

						writer.println(learner.toString());
						writer.println();
						writer.println("Results");
						writer.println("//////////////");
						writer.println(results);
						writer.flush();
						writer.close();

						double m = results.getMean("Macro RelRMSE");
						double d = results.getStd("Macro RelRMSE");

						if (m < bestRelRMSE) {
							bestRelRMSE = m;
							bestDev = d;
							bestH = h;
							bestK = k;
						}

					}
				}

				PrintWriter writer = new PrintWriter(new File("results/" + data + "/GEPMTR-Local/general-result"));

				writer.println("General Results");
				writer.println("//////////////");
				writer.println("Macro RelRMSE: " + bestRelRMSE + "\u00B1" + bestDev);
				writer.println("h: " + bestH);
				writer.println("k: " + bestK);

				writer.flush();
				writer.close();
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
