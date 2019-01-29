package gep;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

import Utils.WriteReadFromFile;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.core.Utils;

public class MyExperiment_GEPMTR {

	public static void main(String args[]) {

		try {

			WriteReadFromFile file = new WriteReadFromFile();

			ArrayList<String> datasets = file.read("configuration/datasets.cfg");
			HashMap<String, int[]> hRanges = file.getRange("configuration/h.cfg");

			int numberOfExecutions = 10;
			int numberOfIndividuals = 50;
			int numberOfGenerations = 100;

			for (String dataset : datasets) {

				double bestRelRMSE = Double.MAX_VALUE;
				double bestDev = 0;
				int bestH = 0;

				String arr[] = dataset.split("\t");
				String data = arr[0];

				int q = Integer.parseInt(arr[1]);

				int range[] = hRanges.get(data);

				for (int h = range[0]; h <= range[1]; h += 5) {

					double means[] = new double[numberOfExecutions];
					double devs[] = new double[numberOfExecutions];

					String path = "results/" + data + "/GEPMTR/h" + h;

					for (int execution = 0; execution < numberOfExecutions; execution++) {

						// Constructing the classifier
						GEPMTRv2 gep = new GEPMTRv2(h, numberOfIndividuals,numberOfGenerations);

						//GEPMTLR learner= new GEPMTLR(h, numberOfIndividuals, numberOfGenerations);
						
						//RAkELMTR learner= new RAkELMTR(gep);

						//EnsembleOfGEPMTR learner = new EnsembleOfGEPMTR(h, numberOfIndividuals, numberOfGenerations, q,
							//	EnsembleOfGEPMTR.SamplingMethod.WithReplacement);

						MultiLabelInstances full = new MultiLabelInstances(
								"datasets/modified/" + data + "-modified.arff", q);

						Evaluator eval = new Evaluator();
						MultipleEvaluation results = null;

						int numFolds = 10;

						results = eval.crossValidate(gep, full, numFolds);

						String pT = path + "-exec" + execution;

						file.createDirectories(path);

						PrintWriter writer = new PrintWriter(new File(pT));

						writer.println(gep.toString());
						writer.println();
						writer.println("Results");
						writer.println("//////////////");
						writer.println(results);
						writer.flush();
						writer.close();

						means[execution] = results.getMean("Macro RelRMSE");
						devs[execution] = results.getStd("Macro RelRMSE");
					}

					double m = Utils.mean(means);
					double d = Utils.mean(devs);

					if (m < bestRelRMSE) {
						bestRelRMSE = m;
						bestDev = d;
						bestH = h;
					}

					PrintWriter writer = new PrintWriter(new File(path));

					writer.println("Results");
					writer.println("//////////////");
					writer.println("Macro RelRMSE: " + m + "\u00B1" + d);
					writer.flush();
					writer.close();
				}

				PrintWriter writer = new PrintWriter(new File("results/" + data + "/GEPMTR/general-result"));

				writer.println("General Results");
				writer.println("//////////////");
				writer.println("Macro RelRMSE: " + bestRelRMSE + "\u00B1" + bestDev);
				writer.println("h: " + bestH);

				writer.flush();
				writer.close();
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
