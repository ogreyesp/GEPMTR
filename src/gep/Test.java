package gep;
import mulan.data.MultiLabelInstances;

public class Test {

	public static void main(String args[]) {

		try {

			int q = 3;

			int numberOfIndividuals = 50;

			int numberGenerations = 100;

			// the length of the head of each gene
			int h = 5;

			MultiLabelInstances mlData = new MultiLabelInstances("datasets/modified/slump-modified.arff", q);

			GEPMTRv2 gep = new GEPMTRv2(h, numberOfIndividuals, numberGenerations);
			
			gep.evaluate(new int[]{3,10,1,9,6,0,2,3,4,3,1}, mlData.getDataSet().instance(0));

			///System.out.println("Fitness:" + gep.bestChromosome.globalFitness);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		/*
		 * int c[] = new int[] { 1, 17, 1, 18, 7, 8, 3, 7, 1, 8, 0, 5, 2, 5, 5,
		 * 7, 0, 2, 8, 1, 8 };
		 * 
		 * // Create a chromosome Chromosome chromo = new Chromosome(c);
		 * 
		 * gep.printChromosome(chromo);
		 * 
		 * gep.printTrees(chromo);
		 * 
		 * c = new int[] { 0, 4, 6, 5, 3, 5, 4, 10, 2, 14, 2, 9, 4, 9, 1, 9, 5,
		 * 5, 5, 5, 5 };
		 * 
		 * // Create a chromosome chromo = new Chromosome(c);
		 * 
		 * gep.printChromosome(chromo);
		 * 
		 * gep.printTrees(chromo);
		 */

	}

}
