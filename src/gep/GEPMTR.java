package gep;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Stack;
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
import weka.core.Utils;

/**
 * Gene Expression Programming for Multi-target Regression.
 * 
 * En esta version consideramos los genes como notaciones prefijas (notacion
 * polaca). Como cada gen tiene una notacion polaca almacenada entonces la
 * evaluacion de un gen puede ser mucho mas rapida que construyendo primero un
 * arbol de expresion y despues evaluar ese arbol. La complejidad computacional
 * respecto a la evaluacion de los individuos se reduce considerablemente.
 * 
 * Esta idea fue propuesta en el articulo:
 * 
 * Peng, Y., Yuan, C., Qin, X., Huang, J., & Shi, Y. (2014). An improved Gene
 * Expression Programming approach for symbolic regression problems.
 * Neurocomputing, 137, 293–301.
 * 
 * @author Oscar Gabriel Reyes Pupo
 *
 */
public class GEPMTR extends MultiLabelLearnerBase {

	private static final long serialVersionUID = 1L;

	// Set of functions
	private String F[] = new String[] {"-", "/", "+", "*", "sroot", "sin","cos","log"};

	// Arity of the functions
	private int arities[] = new int[] {2, 2, 2, 2, 1, 1, 1, 1};

	private int numberOfIndividuals;

	// Population
	private Chromosome[] population;

	// to generate random numbers
	private Random rand;

	// the length of the head
	private int h;

	// the length of the tail
	private int t;

	// length of a gene
	private int lengthGene;

	private int arity = 2;

	private Instances dataset;

	private int numberGenerations = 100;

	// To store the mating pool
	private Chromosome[] matingPool;

	// Typically, a mutation rate equivalent to two mutation points per
	// chromosome is used. The probability of mutation is commonly low;
	private double pm = 0.1;

	// The number of mutation points per chromosome
	private int numberMutationPoints = 2;

	// IS transposition rate
	private double pis = 0.1;

	// Maximum IS elements length
	private int maximumISElementLength = 3;

	// Number of IS elements of different lengths
	private int numberISElements = 3;

	// Root transposition rate
	private double pris = 0.1;

	// Number of RIs elements of different lengths
	private int numberRISElements = 3;

	// Gene transposition rate.
	private double pgt = 0.1;

	// One point recombination rate
	private double por = 0.2;

	// Two point recombination rate
	private double ptr = 0.5;

	// Gene recombination rate
	private double pgr = 0.1;

	// The best chromosome composed by the best genes found across generations
	private Chromosome bestChromosome;
	
	private int tournamentSize = 2;
	
	// Seed for random numbers
	protected long seed;

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

		this.lengthGene = h + t;
		
		this.seed = 1;
	}

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
	 * @param seed
	 *            seed for random numbers
	 *            
	 */
	public GEPMTR(int h, int numberOfIndividuals, int numberGenerations, long seed) {

		this.numberOfIndividuals = numberOfIndividuals;

		this.numberGenerations = numberGenerations;

		this.h = h;

		this.t = h * (arity - 1) + 1;

		this.lengthGene = h + t;
		
		this.seed = seed;
	}
	
	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		
		this.rand = new Random(seed);
		
		this.dataset = trainingSet.getDataSet();

		// Create the initial population
		createInitialPopulation();

		// evaluate the initial population
		doEvaluation(population);

		// do generations
		int iter = 0;

		while (iter < numberGenerations) {

			// Selection
			doSelection();

			// Mutation
			doMutation();

			// Transposition.
			// The transposable elements of GEP are fragments of the genome that
			// can be activated and jump to another place in the chromosome.
			doISTransposition();
			doRISTransposition();

			// Gene Transposition operator can only be used in multi-genic
			// chromosomes
			if (numLabels > 1)
				doGeneTransposition();

			// Recombination
			doOnePointRecombination();
			doTwoPointRecombination();

			// Gene Recombination operator can only be used in multi-genic
			// chromosomes
			if (numLabels > 1)
				doGeneRecombination();

			doEvaluation(matingPool);

			// Prepare new individuals for next generation
			doReplacement();

			iter++;
		}
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {

		// the values predicted
		double[] finalPredictions = new double[numLabels];

		// for each gene
		for (int l = 0; l < numLabels; l++) {

			// copy gene
			int gene[] = new int[lengthGene];

			System.arraycopy(bestChromosome.chromo, l * lengthGene, gene, 0, lengthGene);

			finalPredictions[l] = evaluate(gene, instance);

		}

		MultiLabelOutput mlo = new MultiLabelOutput(finalPredictions, true);
		return mlo;

	}

	// This method creates an initial population. It tries that the individuals
	// will be not similar between each other, guaranteeing the diversity in the
	// initial population. It tries that each gene will be very dissimilar to
	// the exiting ones.
	public void createInitialPopulation() {

		population = new Chromosome[numberOfIndividuals];

		int d = dataset.numAttributes() - numLabels;

		int nCurrent = 0;

		while (nCurrent < numberOfIndividuals) {

			// generate a chromosome
			Chromosome c = new Chromosome(h, t, numLabels, F.length, d, rand);

			boolean check = true;

			for (int i = 0; i < numberOfIndividuals; i++) {

				// check the similarity
				if (population[i] != null) {

					double similarity = similarity(c, population[i]);

					if (similarity > 0.2) {

						//System.out.println("Individual not diverse");
						check = false;
						break;
					}
				} else
					break;
			}

			if (check) {
				population[nCurrent++] = c.copy();
			}
		}
	}

	private void doSelection() {

		doTournamentSelectionCluster();
	}

	/*
	 * private void doTournamentSelectionWithTokenCompetition() {
	 * 
	 * int currentSelection = 0;
	 * 
	 * parents = new Chromosome[numberOfIndividuals];
	 * 
	 * // the 5% of the population is selected int tournamentSize = (int) (0.05
	 * * numberOfIndividuals);
	 * 
	 * while (currentSelection < numberOfIndividuals) {
	 * 
	 * // the parents are selected ArrayList<Integer> selection = new
	 * ArrayList<Integer>();
	 * 
	 * while (selection.size() < tournamentSize) {
	 * 
	 * int sel = rand.nextInt(numberOfIndividuals);
	 * 
	 * while (selection.contains(sel)) sel = rand.nextInt(numberOfIndividuals);
	 * 
	 * selection.add(sel);
	 * 
	 * }
	 * 
	 * // to store the number of tokens wined in each comparison int[]
	 * numberOfTokensWined = new int[tournamentSize];
	 * 
	 * // compare pair of parents for (int i = 0; i < tournamentSize - 1; i++) {
	 * 
	 * Chromosome c1 = population[selection.get(i)];
	 * 
	 * for (int j = i + 1; j < tournamentSize; j++) {
	 * 
	 * Chromosome c2 = population[selection.get(j)];
	 * 
	 * int numberTokens = 0;
	 * 
	 * // for each token for (int l = 0; l < numLabels; l++) { if (c1.fitness[l]
	 * <= c2.fitness[l]) numberTokens++; }
	 * 
	 * numberOfTokensWined[i] += numberTokens; numberOfTokensWined[j] +=
	 * numLabels - numberTokens; } }
	 * 
	 * int maxIndex = Utils.maxIndex(numberOfTokensWined);
	 * 
	 * // put the winner in the next population parents[currentSelection++] =
	 * population[selection.get(maxIndex)].copy(); } }
	 */

	/**
	 * We perform a tournament with replacement. The process is repeated until
	 * the mating pool is filled. This is quite remarkable and says that binary
	 * tournament selection and linear ranking selection are identical in
	 * expectation
	 * 
	 * The selection pressure of tournament selection directly varies with the
	 * tournament size- the more competitors, the higher the resulting selection
	 * pressure. Tournament selection pressure is increased (decreased) by
	 * simply increasing (decreasing) the tournament size.
	 * 
	 * The selection pressure is the degree to which the better individuals are
	 * favored: the higher the selection pressure, the more the better
	 * individuals are favored
	 * 
	 * With higher selection pressures resulting in higher convergence rates. If
	 * the selection pressure is low, the convergence rate will be slow, and the
	 * GA will take longer to find the optimal solution.
	 * 
	 * Increased selection pressure can be provided by simply increasing the
	 * tournament size, as the winner from a large tournament will, on average,
	 * have a higher fitness than the winner of a smaller tournament.
	 */
	/*
	 * private void doTournamentSelection() {
	 * 
	 * int currentSelection = 0;
	 * 
	 * parents = new Chromosome[numberOfIndividuals];
	 * 
	 * // the 5% of the population is selected int tournamentSize = (int) (0.05
	 * * numberOfIndividuals);
	 * 
	 * tournamentSize = 2;
	 * 
	 * while (currentSelection < numberOfIndividuals) {
	 * 
	 * // the parents are selected ArrayList<Integer> selection = new
	 * ArrayList<Integer>();
	 * 
	 * while (selection.size() < tournamentSize) {
	 * 
	 * int sel = rand.nextInt(numberOfIndividuals); selection.add(sel); }
	 * 
	 * int indexBest = -1; double bestFitness = Double.MAX_VALUE;
	 * 
	 * for (int i = 0; i < tournamentSize; i++) { if
	 * (population[selection.get(i)].globalFitness < bestFitness) { bestFitness
	 * = population[selection.get(i)].globalFitness; indexBest = i; } }
	 * 
	 * // put the winner in the next population parents[currentSelection++] =
	 * population[selection.get(indexBest)].copy(); } }
	 */

	/**
	 * Based on the paper: H. Xie and M. Zhang, Tuning selection Pressure in
	 * Tournament Selection.
	 */
	private void doTournamentSelectionCluster() {

		// create the FRD (fitness rank distribution)
		// The clusters are formed by individuals that have the same fitness
		// value

		HashMap<Double, ArrayList<Integer>> frd = new HashMap<Double, ArrayList<Integer>>();

		for (int i = 0; i < numberOfIndividuals; i++) {

			double g = population[i].globalFitness;

			// If the key does not exist
			if (!frd.containsKey(g)) {

				ArrayList<Integer> cluster = new ArrayList<Integer>();
				cluster.add(i);

				frd.put(g, cluster);

			} else { // the key already exists
				frd.get(g).add(i);
			}
		}
// 
		// This variable stores the clusters index (fitness values in our case)
		ArrayList<Double> clusters = new ArrayList<Double>(frd.keySet());

		int currentSelection = 0;

		// The mating pool is created
		matingPool = new Chromosome[numberOfIndividuals];

		while (currentSelection < numberOfIndividuals) {

			int indexBest= rand.nextInt(clusters.size());
			double bestFitness = clusters.get(indexBest);
			
			// Select the winner cluster from the tournament using fitness
			// values

			for (int i = 1; i < tournamentSize; i++) {
				
				int competitor= rand.nextInt(clusters.size());
				
				if (clusters.get(competitor) < bestFitness) {
					bestFitness = clusters.get(competitor);
					indexBest = competitor;
				}
			}

			// Return an individual randomly chosen from the winning cluster

			ArrayList<Integer> clusterWinner = frd.get(clusters.get(indexBest));

			int winner = clusterWinner.get(rand.nextInt(clusterWinner.size()));

			// put the winner in the next population
			matingPool[currentSelection++] = population[winner].copy();
		}
	}

	private void doMutation() {

		int d = dataset.numAttributes() - numLabels;

		ArrayList<Integer> parentsSelected = prepareSelection(pm);

		for (int indexParent : parentsSelected) {

			Chromosome chromosome = matingPool[indexParent];

			for (int i = 0; i < numberMutationPoints; i++) {
				// the gene is selected
				int index = rand.nextInt(numLabels);

				int begin = index * lengthGene;

				int pos = rand.nextInt(lengthGene);

				// determining is the mutation will be in the head or the
				// tail
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

				} else { // in the tails terminals can only change to
							// terminals
					chromosome.chromo[begin + pos] = rand.nextInt(d);
				}
			}
		}
	}

	// Short fragments with a function or terminal in the first position are
	// transposed to the head of genes, except to the root (insertion sequence
	// elements or IS elements).

	// Despite this insertion, the structural organization of chromosomes is
	// maintained, and therefore all newly created individuals are syntactically
	// correct programs. A copy of the transposon is made and inserted at any
	// position in the head of a gene, except at the start position.

	// Transposition can drastically reshape the ET, and the more upstream the
	// insertion site the more profound the change.

	private void doISTransposition() {

		ArrayList<Integer> parentsSelected = prepareSelection(pis);

		for (int indexParent : parentsSelected) {

			Chromosome cro = matingPool[indexParent];

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

					// move the down stream
					System.arraycopy(copyDownStream, 0, targetHead, posTarget + lenIS, cantElementToMove);

				}

				// copy the target head into the original chromosome
				System.arraycopy(targetHead, 0, cro.chromo, targetGene * lengthGene, h);

			}
		}
	}

	// All RIS elements start with a function, and thus are chosen among the
	// sequences of the heads.

	// Despite this insertion, the structural organization of chromosomes is
	// maintained, and therefore all newly created individuals are syntactically
	// correct programs.

	// The modifications caused by root transposition are extremely radical,
	// because the root itself is modified. Like mutation and IS transposition,
	// root insertion
	// has a tremendous transforming power and is excellent for creating genetic
	// variation.

	private void doRISTransposition() {

		ArrayList<Integer> parentsSelected = prepareSelection(pris);

		for (int indexParent : parentsSelected) {

			Chromosome cro = matingPool[indexParent];

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

				// The RIS element is in the range [posBegin,posEnd]
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

	// In gene transposition an entire gene functions as a transposon and
	// transposes itself to the beginning of the chromosome.In contrast to the
	// other forms of transposition, in gene transposition the transposon (the
	// gene) is deleted in the place of origin. This way, the length of the
	// chromosome is maintained.

	private void doGeneTransposition() {

		ArrayList<Integer> parentsSelected = prepareSelection(pgt);

		for (int indexParent : parentsSelected) {

			Chromosome cro = matingPool[indexParent];

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

	// Two parent chromosomes are randomly chosen and paired to exchange some
	// material between them. During one-point recombination, the chromosomes
	// cross over a randomly chosen point to form two chromosomes.
	private void doOnePointRecombination() {

		ArrayList<Integer> parentsSelected = prepareSelection(por);

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

			System.arraycopy(matingPool[posFirst].chromo, 0, chromoFirst, 0, point);
			System.arraycopy(matingPool[posSecond].chromo, point, chromoFirst, point, lengthGene * numLabels - point);

			System.arraycopy(matingPool[posSecond].chromo, 0, chromoSecond, 0, point);
			System.arraycopy(matingPool[posFirst].chromo, point, chromoSecond, point, lengthGene * numLabels - point);

			matingPool[posFirst].chromo = chromoFirst;
			matingPool[posSecond].chromo = chromoSecond;
		}
	}

	// Two parent chromosomes are randomly chosen and paired to exchange some
	// material between them. In two-point recombination the chromosomes are
	// paired and the two points of recombination are randomly chosen. The
	// material between the recombination points is afterwards exchanged
	// between the two chromosomes, forming two new daughter chromosomes.

	// The transforming power of two point recombination is greater than
	// one-point recombination, and is most useful to evolve solutions for more
	// complex problems, especially when multigenic chromosomes composed of
	// several genes are used.

	private void doTwoPointRecombination() {

		ArrayList<Integer> parentsSelected = prepareSelection(ptr);

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

			// pointOne will be in the range [1...n-2]
			int pointOne = 1 + rand.nextInt((lengthGene * numLabels) - 2);
			// pointTwo will be in the range [point+1...n-1]
			int pointTwo = pointOne + 1 + rand.nextInt((lengthGene * numLabels) - pointOne - 1);

			// A complete copy
			System.arraycopy(matingPool[posFirst].chromo, 0, chromoFirst, 0, lengthGene * numLabels);
			// A copy of the range
			System.arraycopy(matingPool[posSecond].chromo, pointOne, chromoFirst, pointOne, (pointTwo - pointOne) + 1);

			// A complete copy
			System.arraycopy(matingPool[posSecond].chromo, 0, chromoSecond, 0, lengthGene * numLabels);
			// A copy of the range
			System.arraycopy(matingPool[posFirst].chromo, pointOne, chromoSecond, pointOne, (pointTwo - pointOne) + 1);

			matingPool[posFirst].chromo = chromoFirst;
			matingPool[posSecond].chromo = chromoSecond;
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
	// variation, more complex problems can only be solved using very large
	// initial populations in order to provide for the necessary diversity of
	// genes.

	private void doGeneRecombination() {

		ArrayList<Integer> parentsSelected = prepareSelection(pgr);

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

			System.arraycopy(matingPool[posFirst].chromo, geneIndex * lengthGene, geneCopy, 0, lengthGene);

			// The gene of the parent2 is copied in parent1
			System.arraycopy(matingPool[posSecond].chromo, geneIndex * lengthGene, matingPool[posFirst].chromo,
					geneIndex * lengthGene, lengthGene);

			// The gene of the parent1 is copied in parent2
			System.arraycopy(geneCopy, 0, matingPool[posSecond].chromo, geneIndex * lengthGene, lengthGene);
		}
	}

	public ArrayList<Integer> prepareSelection(double probability) {

		ArrayList<Integer> parentsSelected = new ArrayList<Integer>();

		for (int i = 0; i < numberOfIndividuals; i++) {

			// participate with certain probability
			if (rand.nextDouble() < probability)
				parentsSelected.add(i);
		}

		return parentsSelected;
	}

	/**
	 * Evaluate an individual
	 */
	public synchronized void evaluate(Chromosome individual) {

		// for each gene
		for (int l = 0; l < numLabels; l++) {

			// copy the gene
			int gene[] = new int[lengthGene];

			System.arraycopy(individual.chromo, l * lengthGene, gene, 0, lengthGene);

			double qerror = 0;

			// For each case
			for (Instance instace : dataset) {

				double predictedValue = evaluate(gene, instace);

				double trueValue = instace.value(labelIndices[l]);

				// Some functions can produce NaN or Infinite
				if (Double.isNaN(predictedValue) || Double.isInfinite(predictedValue)) {
					System.err.println("NaN or infinite");
					System.exit(1);
				}

				qerror += Math.pow(trueValue - predictedValue, 2);
			}

			individual.fitness[l] = Math.sqrt(qerror / dataset.numInstances());
		}

		// The average root mean squared error is used as global fitness
		// function
		individual.globalFitness = Utils.sum(individual.fitness) / numLabels;
	}

	// Evaluate the population of individual passed as argument. The evaluation
	// is done in parallel mode to speed up the evaluation process
	public void doEvaluation(Chromosome[] p) {

		ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

		for (Chromosome cromo : p) {
			threadExecutor.execute(new EvaluationThread(cromo));
		}

		threadExecutor.shutdown();

		try {
			if (!threadExecutor.awaitTermination(30, TimeUnit.DAYS))
				System.out.println("Threadpool timeout occurred");
		} catch (InterruptedException ie) {
			System.out.println("Threadpool prematurely terminated due to interruption in thread");
		}
	}

	public int worstIndividual(Chromosome[] p) {

		double worstFitnessPopulation = Double.MIN_VALUE;

		int worst = 0;

		// The worst individual are determined
		for (int i = 0; i < numberOfIndividuals; i++) {

			double fitness = p[i].globalFitness;

			if (fitness > worstFitnessPopulation) {
				worstFitnessPopulation = fitness;
				worst = i;
			}
		}

		return worst;
	}

	public int bestIndividual(Chromosome[] p) {

		double bestFitnessPopulation = Double.MAX_VALUE;

		int best = 0;
		// The best individuals are determined
		for (int i = 0; i < numberOfIndividuals; i++) {

			double fitness = p[i].globalFitness;

			if (fitness < bestFitnessPopulation) {
				bestFitnessPopulation = fitness;
				best = i;
			}
		}

		return best;
	}

	public Chromosome createBestIndividual(Chromosome p[]) {

		// A chromosome is created from the best genes found in the current
		// population

		Chromosome c = new Chromosome();

		c.chromo = new int[lengthGene * numLabels];
		c.fitness = new double[numLabels];

		// For each label, determine the best gene
		for (int l = 0; l < numLabels; l++) {

			int bestIndex = 0;
			double bestError = Double.MAX_VALUE;

			for (int i = 0; i < numberOfIndividuals; i++) {

				Chromosome cT = p[i];

				if (cT.fitness[l] < bestError) {
					bestError = cT.fitness[l];
					bestIndex = i;
				}
			}

			// Copy the gene of the best individual
			System.arraycopy(p[bestIndex].chromo, l * lengthGene, c.chromo, l * lengthGene, lengthGene);

			c.fitness[l] = p[bestIndex].fitness[l];
		}

		c.globalFitness = Utils.sum(c.fitness) / numLabels;

		return c;
	}

	// This function evaluates the gene (expression in polish notation) with
	// the instance
	public synchronized double evaluate(int gene[], Instance instance) {

		try {

			int eL = computeEffectiveGeneLength(gene);

			Stack<Double> stack = new Stack<Double>();

			for (int index = eL - 1; index >= 0; index--) {

				// it is head
				if (index < h) {

					// if it is a function
					if (gene[index] < F.length) {

						switch (F[gene[index]]) {

						case "+":
							stack.push(stack.pop() + stack.pop());
							break;

						case "-":
							stack.push(stack.pop() - stack.pop());
							break;

						case "/":

							double v1 = stack.pop();
							double v2 = stack.pop();

							// to avoid division by zero
							if (v2 == 0)
								stack.push(0.0);
							else
								stack.push(v1 / v2);

							break;

						case "*":
							stack.push(stack.pop() * stack.pop());
							break;

						case "log":

							double v = stack.pop();

							if (v == 0)
								v = 1e-6;

							// The value cannot be negative
							stack.push(Math.log10(Math.abs(v)));
							break;

						case "sroot":
							// The value cannot be negative
							stack.push(Math.sqrt(Math.abs(stack.pop())));
							break;

						case "sin":
							stack.push(Math.sin(stack.pop()));
							break;

						case "cos":
							stack.push(Math.cos(stack.pop()));
							break;
						}
					} else // it is a terminal
						stack.push(instance.value(gene[index] - F.length));

				} else // it is tail, therefore it is always a terminal
				{
					stack.push(instance.value(gene[index]));
				}
			}

			return stack.pop();

		} catch (Exception e) {
			System.err.println("Malformer expression");
		}

		return Double.NaN;
	}

	/**
	 * This function computes the effective length of the gene, i.e. the code
	 * region of the gene.
	 * 
	 * @param gene
	 *            The gene
	 * @return The effective length of the gene
	 */
	public int computeEffectiveGeneLength(int[] gene) {

		int len = 0;
		int count = 0;

		int index = 0;

		while (count >= 0) {

			len++;

			// it corresponds to the head of the gene
			if (index < h) {

				// it is a function
				if (gene[index] < F.length)
					count = count - 1 + arities[gene[index]];
				else // it is a terminal
					count--;

			} else // it correspond to the tail of the gene
			{
				// it is always a terminal
				count--;
			}

			index++;
		}

		return len;
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

		Chromosome bestOld = createBestIndividual(population);

		int worstIndex = worstIndividual(matingPool);

		// Con esto se garantiza introducir los mejores genes encontrados en la
		// poblacion anterior hacia la nueva generacion
		if (bestOld.globalFitness < matingPool[worstIndex].globalFitness) {

			matingPool[worstIndex] = bestOld.copy();
		}

		// keep in bestChromosome the best chromosome found so far
		if (bestChromosome == null) {
			bestChromosome = bestOld.copy();
		} else {
			// compare the global fitness
			if (bestOld.globalFitness < bestChromosome.globalFitness)
				bestChromosome = bestOld.copy();
		}

		// the generation is replaced by the new one
		population = matingPool;
		matingPool = null;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}

	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Evaluation Thread
	/////////////////////////////////////////////////////////////////

	private class EvaluationThread extends Thread {

		private Chromosome ind;

		public EvaluationThread(Chromosome ind) {
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

	// This method compute the similarity that exists between genes of two
	// chromosomes
	public double similarity(Chromosome c1, Chromosome c2) {

		int total = 0;

		for (int i = 0; i < numLabels; i++) {

			int[] gene1 = new int[lengthGene];
			System.arraycopy(c1.chromo, i * lengthGene, gene1, 0, lengthGene);

			for (int j = 0; j < numLabels; j++) {

				int hamming = 0;

				int[] gene2 = new int[lengthGene];
				System.arraycopy(c2.chromo, j * lengthGene, gene2, 0, lengthGene);

				// count the different elements. Hamming distance
				for (int k = 0; k < lengthGene; k++) {
					if (gene1[k] != gene2[k])
						hamming++;
				}

				total += hamming;
			}
		}		
		return 1 - (total / (double) (numLabels * numLabels * lengthGene));
	}
}
