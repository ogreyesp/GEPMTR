package experiments;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import ensemble.EGEPMTR_S;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import utils.WriteReadFromFile;
import weka.core.Utils;

public class MyExperiment_EGEPMTR_S {

	public static void main(String args[]) {

		try {

			WriteReadFromFile file = new WriteReadFromFile();

			ArrayList<String> datasets = file.read("configuration/datasets.cfg");
			HashMap<String, int[]> hRanges = file.getRange("configuration/h.cfg");

			int numberOfExecutions = 1;
			int numberOfIndividuals = 50;
			int numberOfGenerations = 100;
			int k = 3;

			for (String dataset : datasets) {

				double bestRelRMSE = Double.MAX_VALUE;
				double bestDev = 0;
				int bestH = 0;

				String arr[] = dataset.split("\t");
				String data = arr[0];

				int q = Integer.parseInt(arr[1]);

				int range[] = hRanges.get(data);
				int hStep = 5;

				double [] res = null;

				//For all h values in range with 5-step
				for (int h = range[0]; h <= range[1]; h += hStep) {
					res = runExperimentForH(h, numberOfExecutions, numberOfIndividuals, numberOfGenerations, k, q, data, file, bestRelRMSE);
					if(res != null){
						bestRelRMSE = res[0];
						bestDev = res[1];
						bestH = (int)res[2];
					}
				}
				
				System.out.println("bestH: " + bestH);
				

				FileWriter writerGeneral = new FileWriter(new File("results/" + data + "/EGEPMTRSubset/general-result"), true);
				writerGeneral.write("--------------------\n");
				writerGeneral.flush();
				writerGeneral.close();
			}

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static double[] runExperimentForH(int h, int numberOfExecutions, int numberOfIndividuals, int numberOfGenerations, int k, int q, String data, WriteReadFromFile file, double bestRelRMSE) throws InvalidDataFormatException, IOException{
		double means[] = new double[numberOfExecutions];
		double devs[] = new double[numberOfExecutions];
		double times[] = new double[numberOfExecutions];

		String path = "results/" + data + "/EGEPMTRSubset/h" + h;

		for (int execution = 0; execution < numberOfExecutions; execution++) {
			int seed = execution*10;
			
			// Constructing the classifier
			EGEPMTR_S gep = new EGEPMTR_S(h, k, numberOfIndividuals,numberOfGenerations, seed);

			MultiLabelInstances full = new MultiLabelInstances(
					"datasets/" + data + ".arff", q);
			
			if(k >= full.getNumLabels()){
				System.out.println("k: " + k + " ; numLabels: " + full.getNumLabels());
				return null;
			}

			Evaluator eval = new Evaluator();
			eval.setSeed(seed);
			MultipleEvaluation results = null;

			int numFolds = 10;

			long time_init = System.currentTimeMillis();
			results = eval.crossValidate(gep, full, numFolds);
			long time_end = System.currentTimeMillis();
			times[execution] = ((double)(time_end - time_init)) / numberOfExecutions;
			
			String pT = path + "-exec" + execution;

			file.createDirectories(path);

			PrintWriter writer = new PrintWriter(new File(pT));

			writer.println(((EGEPMTR_S)gep).toString());
			writer.println();
			writer.println("Results");
			writer.println("//////////////");
			writer.println(results);
			writer.println("Average time (ms): " + times[execution]);
			writer.flush();
			writer.close();

			means[execution] = results.getMean("Macro RelRMSE");
			devs[execution] = results.getStd("Macro RelRMSE");
		}

		double m = Utils.mean(means);
		double d = Utils.mean(devs);
		double avgTime = Utils.mean(times);

		double [] best = new double[3];
		if (m < bestRelRMSE) {
			bestRelRMSE = m;
			best[0] = m;
			best[1] = d;
			best[2] = h;
		}
		else{
			best = null;
		}

//		if (m < bestRelRMSE) {
//			bestRelRMSE = m;
//			bestDev = d;
//			bestH = h;
//		}
		
		PrintWriter writer = new PrintWriter(new File(path));

		writer.println("Results");
		writer.println("//////////////");
		writer.println("Macro RelRMSE: " + m + "\u00B1" + d + " ; " + "AvgTime(ms): " + avgTime);
		writer.flush();
		writer.close();
		
		FileWriter writerGeneral = new FileWriter(new File("results/" + data + "/EGEPMTRSubset/general-result"), true);
		writerGeneral.write("h = " + h + " ; " + "Macro RelRMSE: " + m + "\u00B1" + d + " ; " + "AvgTime(ms): " + avgTime + "\n");
		writerGeneral.flush();
		writerGeneral.close();
		
		return best;
	}
}
