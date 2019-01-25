package gep;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 * Gene Expression Programming for Multi-target Regression
 * 
 * @author Oscar Gabriel Reyes Pupo
 *
 */
public class GEPMTR extends MultiLabelLearnerBase {

	private static final long serialVersionUID = 1L;

	public enum SelectionType {
		Roulette, Tournament
	};

	SelectionType selection = SelectionType.Tournament;

	// Set of functions
	String F[] = new String[] { "-", "/", "+", "*", "sroot", "sin", "cos", "log" };

	// Arity of the functions
	int arities[] = new int[] { 2, 2, 2, 2, 1, 1, 1, 1 };

	int numberOfIndividuals;

	// Population
	Chromosome[] population;

	// to generate random numbers
	Random rand;

	// number of elements in the head
	int h;

	// the lenght of the tail
	int t;

	// length of a gene
	int lengthGene;

	int arity = 2;

	Instances dataset;

	int numberGenerations = 100;

	// Computing the mean value of each target variable
	double[] yMeans;

	// To store the parents selected
	Chromosome[] parents;

	// Typically, a mutation rate equivalent to two point mutations per
	// chromosome is used.
	// The probability of mutation is commonly low;
	double pm = 0.1;

	int numberMutationPoints = 2;

	// IS transposition rate (pis) of 0.1 and a set of three IS elements of
	// different length are used.
	double pis = 0.1;

	// Maximum IS elements length
	int maximumISElementLength = 3;

	// Number of IS elements of different lengths
	int numberISElements = 3;

	// Typically a root transposition rate (pris) of 0.1 and a set of three RIs
	// elements of different sizes are used.
	double pris = 0.1;

	// Number of IS elements of different lengths
	int numberRISElements = 3;

	// Gene transposition rate.
	double pgt = 0.1;

	// One point recombination rate
	double por = 0.2;

	// One point recombination rate
	double ptr = 0.5;

	// Gene recombination rate
	double pgr = 0.1;

	// Store the best parent in the last generation
	int bestparent;

	// Store the best offspring in the new generation
	int bestOffspring;

	// Store the worst offspring in the new generation
	int worstOffspring;

	// ensures that best individual passes to the next generation any time

	/**
	 * @param h
	 *            Length of the head in the genes
	 * 
	 * @param ni
	 *            number of individuals
	 * 
	 * @param q
	 *            number of targets
	 * 
	 * @param d
	 *            number of input variable
	 * 
	 */
	public GEPMTR(int h, int numberOfIndividuals, int numberGenerations) {

		this.numberOfIndividuals = numberOfIndividuals;

		this.numberGenerations = numberGenerations;

		this.h = h;

		this.t = h * (arity - 1) + 1;

		lengthGene = h + t;
	}

	public void createInitialPopulation() {

		population = new Chromosome[numberOfIndividuals];

		int d = dataset.numAttributes() - numLabels;

		for (int i = 0; i < numberOfIndividuals; i++) {
			population[i] = new Chromosome(arity, h, numLabels, F.length, d, rand);
		}
	}

	private void doSelection() {

		if (selection == SelectionType.Roulette)
			doRouletteSelection();

		if (selection == SelectionType.Tournament)
			doTournamentSelection();

	}

	private void doRouletteSelection() {

		// prepareSelection()
		double[] roulette = new double[numberOfIndividuals];

		// Sets roulette values
		double acc = 0.0;
		int idx = 0;

		for (Chromosome c : population) {
			acc += 1 / c.globalFitness; // to convert the fitness to maximum
			roulette[idx++] = acc;
		}

		// Normalize roulette values
		for (; idx > 0;) {
			if (acc != 0)
				roulette[--idx] /= acc;
			else
				--idx;
		}

		// a population of equal size of the current is select.
		// Thus, during replication the genomes of the selected individuals are
		// copied as many times as the outcome of the roulette. The roulette is
		// spun as
		// many times as there are individuals in the population, always
		// maintaining the same population size.

		int currentSelection = 0;

		parents = new Chromosome[numberOfIndividuals];

		while (currentSelection < numberOfIndividuals) {

			// Generate a random number

			double p = rand.nextDouble();

			for (int i = 0; i < numberOfIndividuals; i++) {
				if (p < roulette[i]) {
					parents[currentSelection] = population[i].copy();
					break;
				}
			}

			currentSelection++;
		}

	}

	private void doTournamentSelection() {

		int currentSelection = 0;

		parents = new Chromosome[numberOfIndividuals];

		int tournamentSize = 2;

		while (currentSelection < numberOfIndividuals) {

			// Randomly selected individual
			Chromosome winner = population[rand.nextInt(numberOfIndividuals)];

			// Performs tournament
			for (int j = 1; j < tournamentSize; j++) {

				Chromosome opponent = population[rand.nextInt(numberOfIndividuals)];

				if (winner.globalFitness > opponent.globalFitness)
					winner = opponent;
			}

			parents[currentSelection++] = winner.copy();
		}
	}

	private void doMutation() {

		int d = dataset.numAttributes() - numLabels;

		for (Chromosome chromosome : parents) {

			// to perform mutation with certain probability
			double p = rand.nextDouble();

			// do mutation
			if (p < pm) {

				for (int i = 0; i < numberMutationPoints; i++) {
					// the gene is selected
					int index = rand.nextInt(numLabels);

					int begin = index * lengthGene;

					int pos = rand.nextInt(lengthGene);

					// determining is the mutation will be in head or tail
					if (pos < h) {
						// the root can only change for a function
						if (pos == 0)
							chromosome.chromo[begin] = rand.nextInt(F.length);
						else {
							// a coin is launched. The probability of being a
							// terminal or a function is equal.
							int coin = rand.nextInt(2);

							if (coin == 0)
								chromosome.chromo[begin + pos] = rand.nextInt(F.length);
							else
								chromosome.chromo[begin + pos] = F.length + rand.nextInt(d);
						}

					} else { // in the tails terminals can only change into
								// terminals
						chromosome.chromo[begin + pos] = rand.nextInt(d);
					}
				}
			}
		}

	}

	// Despite this insertion, the structural organization of chromosomes is
	// maintained, and therefore all newly created individuals are syntactically
	// correct programs. A copy of the transposon is made and inserted at any
	// position in the head of a gene, except at the start position.
	private void doISTransposition() {

		for (Chromosome cro : parents) {

			double p = rand.nextDouble();

			// do IS transposition
			if (p < pis) {

				// During transposition, the sequence upstream from the
				// insertion site stays unchanged, whereas the sequence
				// downstream from the copied IS element loses, at the end of
				// the head, as many symbols as the length of the IS element

				for (int i = 0; i < numberISElements; i++) {

					int lenIS = 1 + rand.nextInt(maximumISElementLength);

					int pos = rand.nextInt(lengthGene);

					if ((pos + 1) >= lenIS)
						pos -= (lenIS - 1);

					// The IS element will always be between [pos,pos+lenIS]

					// select source gene
					int sourceGene = rand.nextInt(numLabels);

					// select target gene
					int targetGene = rand.nextInt(numLabels);

					// Copy the IS element
					int[] isElement = new int[lenIS];

					System.arraycopy(cro.chromo, sourceGene * lengthGene + pos, isElement, 0, lenIS);

					// Copy the head of the target gene
					int[] targetHead = new int[h];

					System.arraycopy(cro.chromo, targetGene * lengthGene, targetHead, 0, h);

					// A copy of the transposon is made and inserted at any
					// position in the head of a gene, except at the start
					// position

					// A number between 1 and h-1 is generated
					int posTarget = 1 + rand.nextInt(h - 1);

					int cantElementToMove = h - posTarget;

					if (cantElementToMove < lenIS)
						lenIS = cantElementToMove;

					int copyDownStream[] = new int[cantElementToMove];

					System.arraycopy(targetHead, posTarget, copyDownStream, 0, cantElementToMove);

					// copy the IS element in targetHead
					System.arraycopy(isElement, 0, targetHead, posTarget, lenIS);

					if (h > (posTarget + lenIS)) {

						cantElementToMove = h - (posTarget + lenIS);

						// move the down strean
						System.arraycopy(copyDownStream, 0, targetHead, posTarget + lenIS, cantElementToMove);

					}

					// copy the target head into the original chromosome
					System.arraycopy(targetHead, 0, cro.chromo, targetGene * lengthGene, h);

				}

			}
		}
	}

	// Despite this insertion, the structural organization of chromosomes is
	// maintained, and therefore all newly created individuals are syntactically
	// correct programs.

	private void doRISTransposition() {

		for (Chromosome cro : parents) {

			double p = rand.nextDouble();

			// do RIS transposition
			if (p < pris) {

				for (int i = 0; i < numberRISElements; i++) {

					// Select the RIS element

					// All RIS elements start with a function, and thus are
					// chosen among the sequences of the heads.
					// For that, a point is randomly chosen in the head and the
					// gene is scanned downstream until a
					// function is found. This function becomes the start
					// position of the RIS element. If no functions
					// are found, it does nothing.

					// select source gene
					int sourceGene = rand.nextInt(numLabels);

					// Copy the head of the target gene
					int[] targetHead = new int[h];

					System.arraycopy(cro.chromo, sourceGene * lengthGene, targetHead, 0, h);

					// a point is randomly chosen in the head
					int posBegin = rand.nextInt(h);

					int posEnd = posBegin;
					// the gene is scanned downstream until a
					// function is found
					while (targetHead[posBegin] >= F.length) {
						if (posBegin == 0)
							break;
						posBegin--;
					}

					// The RIS element is the range [posBegin,posEnd]
					// Copy the RIS element
					int lenghtRISElement = posEnd - posBegin + 1;
					int[] risElement = new int[lenghtRISElement];

					System.arraycopy(targetHead, posBegin, risElement, 0, lenghtRISElement);

					int[] copyDowmStream = new int[h];

					System.arraycopy(targetHead, 0, copyDowmStream, 0, h);

					// Copy the RIS element in the root of the head
					System.arraycopy(risElement, 0, targetHead, 0, lenghtRISElement);

					// Move the rest of head
					System.arraycopy(copyDowmStream, 0, targetHead, lenghtRISElement, h - lenghtRISElement);

					// copy the target head into the original chromosome
					System.arraycopy(targetHead, 0, cro.chromo, sourceGene * lengthGene, h);

				}

			}
		}
	}

	// In gene transposition an entire gene functions as a transposon and
	// transposes itself to the beginning of the chromosome.In contrast to the
	// other forms of transposition, in gene transposition the transposon (the
	// gene) is deleted in the place of origin. This way, the length of the
	// chromosome is maintained.

	private void doGeneTransposition() {

		for (Chromosome cro : parents) {

			double p = rand.nextDouble();

			// do gene transposition
			if (p < pgt) {

				// The chromosome to undergo gene transposition is randomly
				// chosen, and one of its genes (except the first)
				// is randomly chosen to transpose.

				int geneSource = 1 + rand.nextInt(numLabels - 1);

				int[] chromoCopy = new int[cro.chromo.length];

				// the copy of the first gene
				System.arraycopy(cro.chromo, geneSource * lengthGene, chromoCopy, 0, lengthGene);

				// Copy the rest of the genes in the same order, except the gene
				// that was already copied.

				int index = 1;

				for (int l = 0; l < numLabels; l++) {

					if (l == geneSource)
						continue;

					System.arraycopy(cro.chromo, l * lengthGene, chromoCopy, index * lengthGene, lengthGene);

					index++;

				}

				// Copy the chromosome to the original
				System.arraycopy(chromoCopy, 0, cro.chromo, 0, chromoCopy.length);
			}
		}
	}

	// Two parent chromosomes are randomly chosen and paired to exchange some
	// material between them. During one-point recombination, the chromosomes
	// cross over a randomly chosen point to form two daughter chromosomes.
	private void doOnePointRecombination() {

		ArrayList<Integer> parentsSelected = new ArrayList<Integer>();

		for (int i = 0; i < numberOfIndividuals; i++) {

			double p = rand.nextDouble();
			// participate in the recombination with certain probability
			if (p < por)
				parentsSelected.add(i);
		}

		// check if a even number of parents were selected, otherwise remove the
		// last
		if ((parentsSelected.size() % 2) != 0)
			parentsSelected.remove(parentsSelected.size() - 1);

		int number = parentsSelected.size() - 2;

		for (int i = 0; i <= number; i += 2) {

			int posFirst = parentsSelected.get(i);
			int posSecond = parentsSelected.get(i + 1);

			int chromoFirst[] = new int[lengthGene * numLabels];
			int chromoSecond[] = new int[lengthGene * numLabels];

			// A point between 1..length-2
			int point = 1 + rand.nextInt((lengthGene * numLabels) - 1);

			System.arraycopy(parents[posFirst].chromo, 0, chromoFirst, 0, point);
			System.arraycopy(parents[posSecond].chromo, point, chromoFirst, point, lengthGene * numLabels - point);

			System.arraycopy(parents[posSecond].chromo, 0, chromoSecond, 0, point);
			System.arraycopy(parents[posFirst].chromo, point, chromoSecond, point, lengthGene * numLabels - point);

			parents[posFirst].chromo = chromoFirst;
			parents[posSecond].chromo = chromoSecond;
		}
	}

	// Two parent chromosomes are randomly chosen and paired to exchange some
	// material between them. In two-point recombination the chromosomes are
	// paired and the two points of recombination are randomly chosen. The
	// material between the recombination points is afterwards ex- changed
	// between the two chromosomes, forming two new daughter chromosomes.

	// The transforming power of two point recombination is greater than
	// one-point recombination, and is most useful to evolve solutions for more
	// complex problems, especially when multigenic chromosomes composed of
	// several genes are used.

	private void doTwoPointRecombination() {

		ArrayList<Integer> parentsSelected = new ArrayList<Integer>();

		for (int i = 0; i < numberOfIndividuals; i++) {

			double p = rand.nextDouble();
			// participate in the recombination with certain probability
			if (p < ptr)
				parentsSelected.add(i);
		}

		// check if a even number of parents were selected, otherwise remove the
		// last
		if ((parentsSelected.size() % 2) != 0)
			parentsSelected.remove(parentsSelected.size() - 1);

		int number = parentsSelected.size() - 2;

		for (int i = 0; i <= number; i += 2) {

			// two parents are selected
			int posFirst = parentsSelected.get(i);
			int posSecond = parentsSelected.get(i + 1);

			int chromoFirst[] = new int[lengthGene * numLabels];
			int chromoSecond[] = new int[lengthGene * numLabels];

			int pointOne = 1 + rand.nextInt((lengthGene * numLabels) - 2);
			int pointTwo = pointOne + 1 + rand.nextInt((lengthGene * numLabels) - pointOne - 1);

			// A complete copy
			System.arraycopy(parents[posFirst].chromo, 0, chromoFirst, 0, lengthGene * numLabels);
			// A copy of the range
			System.arraycopy(parents[posSecond].chromo, pointOne, chromoFirst, pointOne, pointTwo - pointOne);

			// A complete copy
			System.arraycopy(parents[posSecond].chromo, 0, chromoSecond, 0, lengthGene * numLabels);
			// A copy of the range
			System.arraycopy(parents[posFirst].chromo, pointOne, chromoSecond, pointOne, pointTwo - pointOne);

			parents[posFirst].chromo = chromoFirst;
			parents[posSecond].chromo = chromoSecond;
		}
	}

	// Two parent chromosomes are randomly chosen and paired to exchange some
	// material between them. In gene recombination an entire gene is exchanged
	// during crossover. The exchanged genes are randomly chosen and occupy the
	// same position in the parent chromosomes.

	// The newly created individuals contain genes from both parents. Note
	// that with this kind of recombination, similar genes can be exchanged but,
	// most of the time, the exchanged genes are very different and new material
	// is introduced into the population.

	// It is worth noting that this operator is unable to create new genes. In
	// fact, when gene recombination is used as the unique source of genetic
	// variation, more complex prob- lems can only be solved using very large
	// initial populations in order to provide for the necessary diversity of
	// genes.

	private void doGeneRecombination() {

		ArrayList<Integer> parentsSelected = new ArrayList<Integer>();

		for (int i = 0; i < numberOfIndividuals; i++) {

			double p = rand.nextDouble();
			// participate in the recombination with certain probability
			if (p < por)
				parentsSelected.add(i);
		}

		// check if a even number of parents were selected, otherwise remove the
		// last
		if ((parentsSelected.size() % 2) != 0)
			parentsSelected.remove(parentsSelected.size() - 1);

		int number = parentsSelected.size() - 2;

		for (int i = 0; i <= number; i += 2) {

			// two parents are selected
			int posFirst = parentsSelected.get(i);
			int posSecond = parentsSelected.get(i + 1);

			// a gene is selected
			int geneIndex = rand.nextInt(numLabels);

			// the gene is interchanged

			int[] geneCopy = new int[lengthGene];

			System.arraycopy(parents[posFirst].chromo, geneIndex * lengthGene, geneCopy, 0, lengthGene);

			System.arraycopy(parents[posSecond].chromo, geneIndex * lengthGene, parents[posFirst].chromo,
					geneIndex * lengthGene, lengthGene);

			System.arraycopy(geneCopy, 0, parents[posSecond].chromo, geneIndex * lengthGene, lengthGene);
		}
	}

	/**
	 * Evaluate an individual
	 */
	public void evaluate(Chromosome individual) {

		double f = 0;

		// for each gene
		for (int l = 0; l < numLabels; l++) {

			// copy gene
			int gene[] = new int[lengthGene];

			System.arraycopy(individual.chromo, l * lengthGene, gene, 0, lengthGene);

			int aritySum[] = computeAritiesSum(gene);

			TreeNode tree = constructTree(gene, 0, aritySum);

			double qerror = 0;

			double ydev = 0;

			// For each case
			for (Instance ins : dataset) {

				double predictedValue = evaluateTree(tree, ins);

				double trueValue = ins.value(labelIndices[l]);

				// Some functions can produce NaN or Infinite
				if (Double.isNaN(predictedValue) || Double.isInfinite(predictedValue)) {
					System.err.println("NaN or infinite");
					System.exit(1);
				}

				qerror += Math.pow(trueValue - predictedValue, 2);
				ydev += Math.pow(trueValue - yMeans[l], 2);
			}

			// the average root mean squared error aRMSE
			f += Math.sqrt(qerror / ydev);
		}

		individual.globalFitness = f / numLabels;
	}

	// Return the index of the best and worst individual in the population
	public int[] doEvaluation(Chromosome[] p) {

		double bestFitnessPopulation = Double.MAX_VALUE;

		double worstFitnessPopulation = Double.MIN_VALUE;

		// bestWorst[0]= index of the best individual
		// bestWorst[1]= index of the worst individual
		int bestWorst[] = new int[2];

		for (int i = 0; i < numberOfIndividuals; i++) {

			evaluate(p[i]);

			double fitness = p[i].globalFitness;

			if (fitness < bestFitnessPopulation) {
				bestFitnessPopulation = fitness;
				bestWorst[0] = i;
			}

			if (fitness > worstFitnessPopulation) {
				worstFitnessPopulation = fitness;
				bestWorst[1] = i;
			}
		}

		return bestWorst;
	}

	// Return the index of the best and worst individual in the population
	public int[] doParallelEvaluation(Chromosome[] p) {

		double bestFitnessPopulation = Double.MAX_VALUE;

		double worstFitnessPopulation = Double.MIN_VALUE;

		// bestWorst[0]= index of the best individual
		// bestWorst[1]= index of the worst individual
		int bestWorst[] = new int[2];

		ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

		for (Chromosome cro : p) {
			threadExecutor.execute(new evaluationThread(cro));
		}

		threadExecutor.shutdown();

		try {
			if (!threadExecutor.awaitTermination(30, TimeUnit.DAYS))
				System.out.println("Threadpool timeout occurred");
		} catch (InterruptedException ie) {
			System.out.println("Threadpool prematurely terminated due to interruption in thread that created pool");
		}

		// The best and worst individuals are determined
		for (int i = 0; i < numberOfIndividuals; i++) {

			double fitness = p[i].globalFitness;

			if (fitness < bestFitnessPopulation) {
				bestFitnessPopulation = fitness;
				bestWorst[0] = i;
			}

			if (fitness > worstFitnessPopulation) {
				worstFitnessPopulation = fitness;
				bestWorst[1] = i;
			}
		}

		return bestWorst;
	}

	// The Expression Tree represented by the gene is evaluated by an inOrder
	// search.
	public double evaluateTree(TreeNode tree, Instance instance) {

		if (tree.isLeaf) {
			return instance.value(tree.index);
		}

		if (!tree.isLeaf) {

			double valueLeft = evaluateTree(tree.left, instance);

			switch (F[tree.index]) {

			case "+":
				return valueLeft + evaluateTree(tree.right, instance);

			case "-":
				return valueLeft - evaluateTree(tree.right, instance);

			case "/":

				double valueRight = evaluateTree(tree.right, instance);

				// to avoid division by zero
				if (valueRight == 0)
					return 0;

				return valueLeft / valueRight;

			case "*":
				return valueLeft * evaluateTree(tree.right, instance);

			case "log":

				if (valueLeft == 0)
					valueLeft = 1e-6;

				// The valueleft cannot be negative
				return Math.log10(Math.abs(valueLeft));

			case "sroot":
				// The valueleft cannot be negative
				return Math.sqrt(Math.abs(valueLeft));

			case "sin":
				return Math.sin(valueLeft);

			case "cos":
				return Math.cos(valueLeft);

			}
		}

		return 0;
	}

	public int[] computeAritiesSum(int[] gene) {

		// compute the sum of arities

		int[] aritySum = new int[h];

		int sum = 0;

		for (int i = 0; i < h; i++) {

			aritySum[i] = sum;

			if (gene[i] < F.length)
				sum += arities[gene[i]];
		}

		return aritySum;
	}

	/**
	 * @param gene
	 *            The gene
	 * @param pos
	 *            The position that is being analysed
	 * 
	 */
	public TreeNode constructTree(int[] gene, int pos, int[] aritySum) {

		// Construct the tree
		TreeNode tree = new TreeNode();

		// verify that it is part of the head
		if (pos < h) {
			// it is a function
			if (gene[pos] < F.length) {
				tree.isLeaf = false;
				tree.index = gene[pos];
			} else // is a terminal symbol
			{
				tree.isLeaf = true;
				tree.index = gene[pos] - F.length;
			}
		} else // it is part of the tail
		{
			tree.isLeaf = true;
			tree.index = gene[pos];
		}

		if (!tree.isLeaf) {

			int currentArity = arities[tree.index];

			// The position of the argument for this function is computed. The
			// argument for the current function are in
			// the range [aritySum+1...aritySum+currentArity]
			int posOne = aritySum[pos] + 1;

			tree.left = constructTree(gene, posOne, aritySum);

			if (currentArity == 2) {
				int posTwo = aritySum[pos] + 2;

				tree.right = constructTree(gene, posTwo, aritySum);
			}

		}

		return tree;
	}

	// Print the tree with the Breadth First Search algorithm
	public void printTree(TreeNode tree) {

		Queue<TreeNode> queue = new LinkedList<TreeNode>();

		queue.add(tree);

		StringBuilder str = new StringBuilder();

		while (!queue.isEmpty()) {

			TreeNode nodeT = queue.poll();

			// It is head
			if (nodeT.isLeaf)
				str.append(" x" + nodeT.index);
			else {
				str.append(" " + F[nodeT.index]);
			}

			if (nodeT.left != null)
				queue.add(nodeT.left);
			if (nodeT.right != null)
				queue.add(nodeT.right);
		}

		System.out.println(str);
	}

	/**
	 * Evaluate an individual
	 */
	public void printTrees(Chromosome individual) {

		// for each gene
		for (int l = 0; l < numLabels; l++) {

			System.out.println("Gene " + l + ":");

			// copy gene
			int gene[] = new int[lengthGene];
			System.arraycopy(individual.chromo, l * lengthGene, gene, 0, lengthGene);

			int[] aritySum = computeAritiesSum(gene);

			TreeNode tree = constructTree(gene, 0, aritySum);

			printTree(tree);

		}

	}

	/**
	 * Print a chromosome
	 * 
	 * @param individualIndex
	 */
	public void printChromosome(Chromosome individual) {

		System.out.println("Chromosome ");

		int[] chromo = individual.chromo;

		// for each gene
		for (int i = 0; i < numLabels; i++) {

			System.out.println("Gene " + i + ":");

			int begin = i * lengthGene;
			int end = i * lengthGene + lengthGene;

			StringBuilder head = new StringBuilder("Head:");
			StringBuilder tail = new StringBuilder("Tail:");

			for (int j = begin; j < end; j++) {

				// check if head
				if ((j - begin) < h) {
					// check if the symbol if a function
					if (chromo[j] < F.length)
						head.append(" " + F[chromo[j]]);
					else // is a terminal symbol
						head.append(" x" + (chromo[j] - F.length));
				} else // is tail
					tail.append(" x" + chromo[j]);
			}
			System.out.println(head);
			System.out.println(tail);
		}

		System.out.println();

	}

	// Like a generational algorithm,
	// but ensures that best individual passes to the next generation any time.
	private void doReplacement() {

		// If best individual in population set is better that best individual
		// in parents set, remove worst individual in parents set and add the
		// best to parents set

		// To do a copy of the best in the new population
		if (population[bestparent].globalFitness < parents[bestOffspring].globalFitness) {
			parents[worstOffspring] = population[bestparent].copy();
			bestparent = worstOffspring;
		} else {
			bestparent = bestOffspring;
		}

		// the generation is replaced is replaced by the new one
		population = parents;
	}

	public Chromosome getBestSolution() {
		return population[bestparent];
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		
		rand = new Random();
		
		this.dataset = trainingSet.getDataSet();

		// Computing the mean value of each target variable
		yMeans = new double[numLabels];

		// for each label
		for (int l = 0; l < numLabels; l++) {
			yMeans[l] = dataset.meanOrMode(labelIndices[l]);
		}

		// Create the initial population
		createInitialPopulation();

		// evaluate the initial population
		bestparent = doParallelEvaluation(population)[0];

		// do generations
		int iter = 0;

		while (iter < numberGenerations) {

			doSelection();
			doMutation();
			doISTransposition();
			doRISTransposition();

			// Gene Transposition operator can only be used in multi-genic
			// chromosomes
			if (numLabels > 1)
				doGeneTransposition();

			doOnePointRecombination();
			doTwoPointRecombination();

			// Gene Recombination operator can only be used in multi-genic
			// chromosomes
			if (numLabels > 1)
				doGeneRecombination();

			// The best and worst individuals of the new population are
			// retrieved
			int[] arr = doParallelEvaluation(parents);

			bestOffspring = arr[0];
			worstOffspring = arr[1];

			doReplacement();

			iter++;
		}

		if (getDebug()) {
			// printing the best regressor found
			printTrees(population[bestparent]);

			System.out.println("Fitness: " + population[bestparent].globalFitness);
		}
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {

		// the values predicted
		double[] finalPredictions = new double[numLabels];

		Chromosome bestEstimator = population[bestparent];

		// for each gene
		for (int l = 0; l < numLabels; l++) {

			// copy gene
			int gene[] = new int[lengthGene];

			System.arraycopy(bestEstimator.chromo, l * lengthGene, gene, 0, lengthGene);

			int aritySum[] = computeAritiesSum(gene);

			TreeNode tree = constructTree(gene, 0, aritySum);

			finalPredictions[l] = evaluateTree(tree, instance);

		}

		MultiLabelOutput mlo = new MultiLabelOutput(finalPredictions, true);
		return mlo;

	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}

	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Evaluation Thread
	/////////////////////////////////////////////////////////////////

	private class evaluationThread extends Thread {
		private Chromosome ind;

		public evaluationThread(Chromosome ind) {
			this.ind = ind;
		}

		public void run() {
			evaluate(ind);
		}
	}

	public String toString() {

		StringBuilder str = new StringBuilder();

		str.append("Number of individuals:" + numberOfIndividuals).append("\n");
		str.append("Number of generations:" + numberGenerations).append("\n");
		str.append("Length of the head of genes:" + h).append("\n");
		str.append("Mutation probability:" + pm).append("\n");
		str.append("Number of mutation points per chromosome:" + numberMutationPoints).append("\n");
		str.append("Probability of IS Transposition:" + pis).append("\n");
		str.append("Maximum length of IS elements:" + maximumISElementLength).append("\n");
		str.append("Number of IS elements:" + numberISElements).append("\n");
		str.append("Probability of RIS Transposition:" + pris).append("\n");
		str.append("Number of RIS elements:" + numberRISElements).append("\n");
		str.append("Probability of Gene Transposition:" + pgt).append("\n");
		str.append("Probability of One Point Recombination:" + por).append("\n");
		str.append("Probability of Two-Point Recombination:" + ptr).append("\n");
		str.append("Probability of Gene Recombination:" + pgr).append("\n");

		return str.toString();
	}
}
