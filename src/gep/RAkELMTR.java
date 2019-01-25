package gep;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class implementing a version of the RAkEL (RAndom k-labELsets) algorithm for
 * multi-target regression.
 *
 * @author Oscar Gabriel Reyes Pupo
 * @version 2016.07
 */
@SuppressWarnings("serial")
public class RAkELMTR extends MultiLabelMetaLearner {

	/**
	 * Random number generator
	 */
	private Random rnd;
	double[][] sumVotesIncremental;

	double[][] lengthVotesIncremental;
	double[] sumVotes;
	double[] lengthVotes;

	static int numOfModels;

	int sizeOfSubset = 3;
	int[][] classIndicesPerSubset;
	int[][] absoluteIndicesToRemove;

	MultiLabelLearner[] subsetClassifiers;

	private Remove[] remove;

	HashSet<String> combinations;

	// Always an extra classifier with all targets is trained
	private GEPMTRv2 gep;

	@Override
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}

	/**
	 * Creates an instance based on a given multi-label learner
	 * 
	 * @param baseLearner
	 *            the multi-label learner
	 */
	public RAkELMTR(MultiLabelLearner baseLearner) {
		super(baseLearner);
	}

	/**
	 * Creates an instance given a specific multi-label learner, number of
	 * models and size of subsets
	 * 
	 * @param baseLearner
	 *            a multi-label learner
	 * @param models
	 *            a number of models
	 * @param subset
	 *            a size of subsets
	 */
	public RAkELMTR(MultiLabelLearner baseLearner, int models, int subset) {
		super(baseLearner);
		sizeOfSubset = subset;
		numOfModels = models;
	}

	/**
	 * Sets the size of the subsets
	 * 
	 * @param size
	 *            the size of the subsets
	 */
	public void setSizeOfSubset(int size) {
		sizeOfSubset = size;
		classIndicesPerSubset = new int[numOfModels][sizeOfSubset];
	}

	/**
	 * Returns the size of the subsets
	 * 
	 * @return the size of the subsets
	 */
	public int getSizeOfSubset() {
		return sizeOfSubset;
	}

	/**
	 * Sets the number of models
	 * 
	 * @param models
	 *            number of models
	 */
	public void setNumModels(int models) {
		numOfModels = models;
	}

	/**
	 * Returns the number of models
	 * 
	 * @return number of models
	 */
	public int getNumModels() {
		return numOfModels;
	}

	/**
	 * The binomial function
	 * 
	 * @param n
	 *            Binomial coefficient index
	 * @param m
	 *            Binomial coefficient index
	 * @return The result of the binomial function
	 */
	public static int binomial(int n, int m) {
		int[] b = new int[n + 1];
		b[0] = 1;
		for (int i = 1; i <= n; i++) {
			b[i] = 1;
			for (int j = i - 1; j > 0; --j) {
				b[j] += b[j - 1];
			}
		}
		return b[m];
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainingData) throws Exception {

		rnd = new Random();

		// need a structure to hold different combinations
		combinations = new HashSet<String>();
		// MultiLabelInstances mlDataSet = trainData.clone();

		// check whether sizeOfSubset is larger or equal compared to number of
		// labels
		if (numLabels == 3) {
			sizeOfSubset = 2;
		}

		if (sizeOfSubset >= numLabels) {
			throw new IllegalArgumentException("Size of subsets should be less than the number of labels");
		}

		// default number of models = twice the number of labels
		if (numOfModels == 0) {

			numOfModels = Math.min(2 * numLabels, binomial(numLabels, sizeOfSubset));
		}

		classIndicesPerSubset = new int[numOfModels][sizeOfSubset];
		absoluteIndicesToRemove = new int[numOfModels][sizeOfSubset];
		subsetClassifiers = new MultiLabelLearner[numOfModels];

		remove = new Remove[numOfModels];

		for (int i = 0; i < numOfModels; i++) {
			updateClassifier(trainingData, i);
		}

		gep = (GEPMTRv2) getBaseLearner().makeCopy();
		gep.build(trainingData);
	}

	/*
	 * private void determineCorrelations(Instances dataset) { try {
	 * 
	 * CorrelationAttributeEval corr = new CorrelationAttributeEval();
	 * 
	 * for (int i = 0; i < numLabels - 1; i++) {
	 * 
	 * dataset.setClassIndex(labelIndices[i]); corr.buildEvaluator(dataset);
	 * 
	 * for (int l = i + 1; l < numLabels; l++) {
	 * System.out.println("Correlation "+i+","+l+": "+
	 * corr.evaluateAttribute(labelIndices[l])); } }
	 * 
	 * } catch (Exception e) { // TODO Auto-generated catch block
	 * e.printStackTrace(); }
	 * 
	 * }
	 */

	private void updateClassifier(MultiLabelInstances mlTrainData, int model) throws Exception {

		Instances trainData = mlTrainData.getDataSet();

		// select a random subset of classes not seen before
		boolean[] selected;

		do {
			selected = new boolean[numLabels];

			for (int j = 0; j < sizeOfSubset; j++) {

				int randomLabel = rnd.nextInt(numLabels);
				
				while (selected[randomLabel] != false) {
					randomLabel = rnd.nextInt(numLabels);
				}
				
				selected[randomLabel] = true;
				
				// System.out.println("label: " + randomLabel);
				classIndicesPerSubset[model][j] = randomLabel;
			}

			Arrays.sort(classIndicesPerSubset[model]);
		} while (combinations.add(Arrays.toString(classIndicesPerSubset[model])) == false);

		debug("Building model " + (model + 1) + "/" + numOfModels + ", subset: "
				+ Arrays.toString(classIndicesPerSubset[model]));

		// remove the unselected labels
		absoluteIndicesToRemove[model] = new int[numLabels - sizeOfSubset];
		int k = 0;
		for (int j = 0; j < numLabels; j++) {
			if (selected[j] == false) {
				absoluteIndicesToRemove[model][k] = labelIndices[j];
				k++;
			}
		}
		remove[model] = new Remove();
		remove[model].setAttributeIndicesArray(absoluteIndicesToRemove[model]);
		remove[model].setInputFormat(trainData);
		remove[model].setInvertSelection(false);
		Instances trainSubset = Filter.useFilter(trainData, remove[model]);

		// build a MultiLabelLearner for the selected label subset;
		subsetClassifiers[model] = getBaseLearner().makeCopy();
		subsetClassifiers[model].build(mlTrainData.reintegrateModifiedDataSet(trainSubset));
	}

	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
		double[] sumPrediction = new double[numLabels];
		lengthVotes = new double[numLabels];

		// gather votes
		for (int i = 0; i < numOfModels; i++) {
			remove[i].input(instance);
			remove[i].batchFinished();
			Instance newInstance = remove[i].output();
			MultiLabelOutput subsetMLO = subsetClassifiers[i].makePrediction(newInstance);

			for (int j = 0; j < sizeOfSubset; j++) {
				sumPrediction[classIndicesPerSubset[i][j]] += subsetMLO.getPvalues()[j];
				lengthVotes[classIndicesPerSubset[i][j]]++;
			}
		}

		double[] pValues = gep.makePrediction(instance).getPvalues();

		for (int i = 0; i < numLabels; i++) {
			sumPrediction[i] += pValues[i];
			lengthVotes[i]++;
		}

		for (int i = 0; i < numLabels; i++) {
			if (lengthVotes[i] != 0) {
				sumPrediction[i] = sumPrediction[i] / lengthVotes[i];
			}
		}

		// todo: optionally use confidence2 for ranking measures
		MultiLabelOutput mlo = new MultiLabelOutput(sumPrediction, true);
		return mlo;
	}

	/**
	 * Returns a string describing classifier
	 *
	 * @return a description suitable for displaying
	 */
	public String globalInfo() {
		return "Class implementing a generalized version of the RAkEL "
				+ "(RAndom k-labELsets) algorithm. For more information, see\n\n"
				+ getTechnicalInformation().toString();
	}

	public String toString() {

		StringBuilder str = new StringBuilder();

		str.append(baseLearner.toString());
		str.append("Number of models:" + numOfModels).append("\n");
		str.append("Size of subsets:" + sizeOfSubset).append("\n");

		return str.toString();
	}
}