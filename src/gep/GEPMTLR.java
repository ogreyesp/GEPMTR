package gep;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MultiLabelKNN;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 * Gene Expression Programming for Local Multi-target Regression.
 * 
 * @author Oscar Reyes
 *
 */
public class GEPMTLR extends MultiLabelKNN {

	private static final long serialVersionUID = 1L;

	private int h;
	private int numberOfIndividuals;
	private int numberGenerations;
	private LabelsMetaData metaData;
	private MultiLabelLearner learner;
	private static int numL = 0;

	public GEPMTLR(int h, int numberOfIndividuals, int numberGenerations, int k) {

		super(k);
		this.h = h;
		this.numberOfIndividuals = numberOfIndividuals;
		this.numberGenerations = numberGenerations;
	}

	protected void buildInternal(MultiLabelInstances trainSet) throws Exception {

		super.buildInternal(trainSet);

		metaData = trainSet.getLabelsMetaData();
		
		if(numL==0)
			numL= numLabels;
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {

		Instances instances = lnn.kNearestNeighbours(instance, numOfNeighbors);

		GEPMTRv2 gep = new GEPMTRv2(h, numberOfIndividuals, numberGenerations);

		if (numLabels == 2) {
			learner = gep;
		} else {

			learner = new RAkELMTR(gep);
		}

		learner.build(new MultiLabelInstances(instances, metaData));
		return learner.makePrediction(instance);

	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}

	public String toString() {

		GEPMTRv2 gep = new GEPMTRv2(h, numberOfIndividuals, numberGenerations);

		if (numL == 2) {
			learner = gep;
		} else {
			learner = new RAkELMTR(gep);
		}

		StringBuilder str = new StringBuilder();

		str.append(learner.toString());
		str.append("Number of neighbors:" + numOfNeighbors).append("\n");

		return str.toString();
	}
}
