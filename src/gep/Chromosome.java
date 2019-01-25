package gep;

import java.util.Random;

public class Chromosome implements Comparable<Chromosome> {

	// The chromosome
	int[] chromo;

	// The global fitness
	double globalFitness;

	// It stores the fitness for each target
	double[] fitness;

	/**
	 * @param head
	 *            The length of the head
	 * @param tail
	 *            The length of the tail
	 * @param genes
	 *            The number of genes
	 * @param f
	 *            The number of functions
	 * @param t
	 *            The number of terminals
	 * @param rand
	 *            The object to generate random numbers
	 */
	public Chromosome(int head, int tail, int genes, int f, int t, Random rand) {

		// length of a gene
		int length = head + tail;

		chromo = new int[genes * length];

		fitness = new double[genes];

		// Create the chromosome

		// for each gene
		for (int i = 0; i < genes; i++) {

			int begin = i * length;
			int end = i * length + length;

			for (int j = begin; j < end; j++) {

				// check if is head
				if ((j - begin) < head) {
					// avoid that the root will be a terminal symbol
					if ((j - begin) == 0)
						chromo[j] = rand.nextInt(f);
					else {
						// a coin is launched. The probability of being a
						// terminal or a function is equal.
						int coin = rand.nextInt(2);

						if (coin == 0) //a function
							chromo[j] = rand.nextInt(f);
						else //a terminal
							chromo[j] = f + rand.nextInt(t);
					}

				} else // is tail
					chromo[j] = rand.nextInt(t);
			}
		}
	}

	public Chromosome() {

	}

	public Chromosome copy() {

		Chromosome co = new Chromosome();

		co.chromo = chromo.clone();
		co.globalFitness = globalFitness;
		co.fitness = fitness.clone();

		return co;
	}

	@Override
	public int compareTo(Chromosome o) {

		if (globalFitness < o.globalFitness)
			return -1;

		if (globalFitness > o.globalFitness)
			return 1;

		return 0;
	}
}
