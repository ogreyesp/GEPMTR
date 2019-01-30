package ensemble;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.Stack;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import gep.GEPMTR;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelsMetaData;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import utils.WriteReadFromFile;
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
public class EGEPMTR_S extends MultiLabelLearnerBase {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4584565708106516712L;

	protected int h;
	
	protected int numberOfIndividuals;
	
	protected int numberGenerations;
	
	protected int seed;
	
	protected int numberOfModels;
	
	protected int k;
	
	Random rand;
	
	byte [][] subsetsMatrix;
	
	int [] votesPerLabel;

	// Full targets regressor
	public GEPMTR fullTargetsRegressor;
	
	// Rest of the ensemble
	public GEPMTR[] ensemble;

	
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
	public EGEPMTR_S(int h, int k, int numIndividuals, int numGenerations, int seed) {
		this.h = h;
		this.k = k;
		this.numberOfIndividuals = numIndividuals;
		this.numberGenerations = numGenerations;
		this.seed = seed;
		
//		numberOfModels = numLabels + 1;
	}


	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		
		this.rand = new Random(seed);
		
		int avgExpectedVotes = 6;
		numLabels = trainingSet.getNumLabels();
//		numberOfModels = numLabels + 1;
		numberOfModels = (int)Math.ceil((avgExpectedVotes / (double)k) * numLabels) + 1;
//		System.out.println(numberOfModels);
		
		votesPerLabel = new int[numLabels];
		
		subsetsMatrix = new byte[numberOfModels-1][numLabels];
		
		fullTargetsRegressor = new GEPMTR(h, numberGenerations, numberOfIndividuals, seed);
		fullTargetsRegressor.build(trainingSet);
		
		for(int i=0; i<numLabels; i++){
			votesPerLabel[i] = 1;
		}
		
		ensemble = new GEPMTR[numberOfModels-1];
		for(int n=0; n<(numberOfModels-1); n++){
			//Select random subset
			byte [] subset = selectRandomSubset(k, numLabels);
			System.arraycopy(subset, 0, subsetsMatrix[n], 0, numLabels);
			
			//Update votesPerLabel
			for(int i=0; i<numLabels; i++){
				if(subset[i] == 1){
					votesPerLabel[i]++;
				}
			}
			
			//Generate ensemble member
			ensemble[n] = new GEPMTR(h, numberGenerations, numberOfIndividuals, seed*n);
			ensemble[n].build(transformInstances(trainingSet, subsetsMatrix[n]));
		}
	}
	
	protected byte [] selectRandomSubset(int k, int numLabels){
		byte [] subset = new byte[numLabels];
		
		int i=0;
		int r;
		
		while(i < k){
			r = rand.nextInt(numLabels);
			if(subset[r] == 0){
				subset[r] = 1;
				i++;
			}
		}
		
		return subset;
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {	
		// Final predicted values
		double[] finalPredictions = new double[numLabels];
		
		MultiLabelOutput mlo = fullTargetsRegressor.makePrediction((Instance)instance.copy());
		System.arraycopy(mlo.getPvalues(), 0, finalPredictions, 0, numLabels);
		
		Instance modifiedInst = transformInstance(instance);
		for(int n=0; n<(numberOfModels-1); n++){
			mlo = ensemble[n].makePrediction(modifiedInst);

			for (int label=0, k=0; label < numLabels; label++) {   
				if(subsetsMatrix[n][label] == 1){
					finalPredictions[label] += mlo.getPvalues()[k];
		            k++;
				}
		    }
		}
		
		for(int n=0; n<numLabels; n++){
			finalPredictions[n] /= ((double)votesPerLabel[n]);
		}
		
		MultiLabelOutput finalMLO = new MultiLabelOutput(finalPredictions, true);
		return finalMLO;
	}
	
	
	protected MultiLabelInstances transformInstances(MultiLabelInstances mlData, byte [] subset) 
			throws InvalidDataFormatException{
		
		labelNames = mlData.getLabelNames();
		
		//Get current LabelsMetaData
		LabelsMetaDataImpl lMeta = (LabelsMetaDataImpl) mlData.getLabelsMetaData().clone();
		
		for(int i=0; i<subset.length; i++){
			if(subset[i] == 0){
				lMeta.removeLabelNode(labelNames[i]);
			}
		}

		MultiLabelInstances transformed = new MultiLabelInstances(mlData.clone().getDataSet(), lMeta);
		
		return(transformed);
	}
	
	protected Instance transformInstance(Instance instance, int label){
		
		Instance extInstance = (Instance) instance.copy();
		
		try {
			MultiLabelOutput predict = fullTargetsRegressor.makePrediction(extInstance);
			
			extInstance.setValue(labelIndices[label], predict.getPvalues()[label]);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
//		System.out.println("ext l:" + label + " ; " + extInstance.toString());

		return(extInstance);
	}
	
	protected Instance transformInstance(Instance instance){
		
		Instance extInstance = (Instance) instance.copy();
		
		try {
			MultiLabelOutput predict = fullTargetsRegressor.makePrediction(extInstance);
			
			for(int label=0; label<numLabels; label++){
				extInstance.setValue(labelIndices[label], predict.getPvalues()[label]);
			}			
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
//		System.out.println("ext l:" + label + " ; " + extInstance.toString());

		return(extInstance);
	}
	
	private MultiLabelInstances bagging(MultiLabelInstances mlData){
    	
    	Instances baggInstances = new Instances(mlData.clone().getDataSet());
    	baggInstances.clear();
    	Instances data = mlData.getDataSet();
    	
    	int nInst = mlData.getNumInstances();
    	
    	int r;
    	for(int i=0; i<nInst; i++){
    		r = rand.nextInt(nInst);
    		baggInstances.add(data.get(r));
    	}
    	
    	MultiLabelInstances baggData = null;
		try {
			baggData = mlData.reintegrateModifiedDataSet(baggInstances);
		} catch (InvalidDataFormatException e) {
			e.printStackTrace();
		}
    	
    	return baggData;
    }

	public String toString() {

		StringBuilder str = new StringBuilder();

		str.append("Number of individuals:" + numberOfIndividuals).append("\n");
		str.append("Number of generations:" + numberGenerations).append("\n");
		str.append("Length of the head of genes:" + h).append("\n");
		str.append("Subset size:" + k).append("\n");
		str.append("Number of models:" + numberOfModels).append("\n");

		return str.toString();
	}


	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	
}
