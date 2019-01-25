package gep;
import java.util.Random;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

/**
 * This class implements the Ensemble of Gene Expression Programming for
 * Multi-target Regression
 * 
 * Oscar Gabriel Reyes Pupo
 */
public class EnsembleOfGEPMTR extends MultiLabelLearnerBase {

	private static final long serialVersionUID = 1L;

	/**
	 * The number of GEPMTR models to be created.
	 */
	private int numOfModels;

	/**
	 * Stores the GEPMTR models.
	 */
	private GEPMTRv2[] ensemble;

	/**
	 * Three types of sampling.
	 */
	public enum SamplingMethod {
		None, WithReplacement, WithoutReplacement,
	};

	/**
	 * The type of sampling to be used. None is used by default.
	 */
	private SamplingMethod sampling = SamplingMethod.None;

	/**
	 * The size of each sample (as a percentage of the training set size) when
	 * sampling with replacement is performed. Default is 100.
	 */
	private double sampleWithReplacementPercent = 100;

	/**
	 * The size of each sample (as a percentage of the training set size) when
	 * sampling without replacement is performed. Default is 67.
	 */
	private double sampleWithoutReplacementPercent = 67;

	// length of the head of the genes
	private int h;

	private int numberOfIndividuals;

	private int numberOfGenerations;

	/**
	 * Constructor.
	 * 
	 * @param baseRegressor
	 *            the base regression algorithm that will be used
	 * @param numOfModels
	 *            the number of models in the ensemble
	 * @param sampling
	 *            the sampling method
	 * @throws Exception
	 *             Potential exception thrown. To be handled in an upper level.
	 */
	public EnsembleOfGEPMTR(int h, int numberOfIndividuals, int numberGenerations, int numOfModels, SamplingMethod sampling)
			throws Exception {

		this.h = h;
		this.numberOfIndividuals = numberOfIndividuals;
		this.numberOfGenerations = numberGenerations;

		this.sampling = sampling;
		this.numOfModels = numOfModels;

		ensemble = new GEPMTRv2[numOfModels];
	}

	@Override
	protected void buildInternal(MultiLabelInstances mlTrainSet) throws Exception {

		// this random number generator will be used for taking random samples
		// and creating random chains
		Random rand = new Random();

		for (int i = 0; i < numOfModels; i++) {

			debug("Building Model:" + (i + 1) + "/" + numOfModels);

			MultiLabelInstances sampledTrainingSet;

			if (sampling != SamplingMethod.None) {

				// initialize a Resample filter using a different seed each time

				Resample rsmp = new Resample();
				rsmp.setRandomSeed(rand.nextInt());

				if (sampling == SamplingMethod.WithoutReplacement) {
					rsmp.setNoReplacement(true);
					rsmp.setSampleSizePercent(sampleWithoutReplacementPercent);
				} else {
					rsmp.setNoReplacement(false);
					rsmp.setSampleSizePercent(sampleWithReplacementPercent);
				}

				Instances trainSet = new Instances(mlTrainSet.getDataSet());
				rsmp.setInputFormat(trainSet);

				Instances sampled = Filter.useFilter(trainSet, rsmp);
				sampledTrainingSet = new MultiLabelInstances(sampled, mlTrainSet.getLabelsMetaData());
			} else {
				sampledTrainingSet = mlTrainSet;
			}

			ensemble[i] = new GEPMTRv2(h, numberOfIndividuals, numberOfGenerations);

			ensemble[i].build(sampledTrainingSet);
		}
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {

		double[] scores = new double[numLabels];

		for (int i = 0; i < numOfModels; i++) {

			MultiLabelOutput ensembleMLO = ensemble[i].makePrediction(instance);

			double[] score = ensembleMLO.getPvalues();

			for (int j = 0; j < numLabels; j++) {
				scores[j] += score[j];
			}
		}

		for (int j = 0; j < numLabels; j++) {
			scores[j] /= numOfModels;
		}

		MultiLabelOutput mlo = new MultiLabelOutput(scores, true);
		return mlo;
	}

	public void setSampleWithReplacementPercent(int sampleWithReplacementPercent) {
		this.sampleWithReplacementPercent = sampleWithReplacementPercent;
	}

	public void setSampleWithoutReplacementPercent(double sampleWithoutReplacementPercent) {
		this.sampleWithoutReplacementPercent = sampleWithoutReplacementPercent;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}
	
	public String toString() {

		StringBuilder str = new StringBuilder();

		str.append(new GEPMTR(h, numberOfIndividuals, numberOfGenerations).toString());
		str.append("Number of models:"+ numOfModels).append("\n");
		str.append("Sampling:"+ sampling).append("\n");
		
		return str.toString();
	}
}
