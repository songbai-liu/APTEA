package jmetal.metaheuristics.aptea;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import jmetal.core.Algorithm;
import jmetal.core.Operator;
import jmetal.core.Problem;
import jmetal.core.Solution;
import jmetal.core.SolutionSet;
import jmetal.core.Variable;
import jmetal.metaheuristics.moead.Utils;
import jmetal.qualityIndicator.GenerationalDistance;
import jmetal.util.JMException;
import jmetal.util.PseudoRandom;
import jmetal.util.ranking.NondominatedRanking;
import jmetal.util.ranking.Ranking;
import jmetal.util.vector.TwoLevelWeightVectorGenerator;
import jmetal.util.vector.VectorGenerator;
import jmetal.util.wrapper.XReal;

public class LAPTEA2 extends Algorithm{
	
	private int populationSize;//population size
	private int numObj_; //number of objectives
	private int numVar_; //number of variables
	
	/**
	 * Stores the population
	 */
	private SolutionSet previousPopulation_;
	private SolutionSet population_;
	private SolutionSet offspringPopulation_;
	private SolutionSet unionPopulation_;	
	private int evaluations;
	private int maxEvaluations;	
	/**
	 * Operators
	 */
	private Operator crossoverOperator_;
	private Operator mutationOperator_;	
	private int groupSize_; //number of groups
	private int[][] group_;
	double[] zideal_;
	double[] nadir_;
	double[] UB_;
	double[] LB_;
	
	GenerationalDistance gd_;
	
	public LAPTEA2(Problem problem) {
		super(problem);
		numObj_ = problem.getNumberOfObjectives();
		numVar_ = problem.getNumberOfVariables();
	}

	public SolutionSet execute() throws JMException, ClassNotFoundException {
		initialization();
		while(evaluations <= maxEvaluations){
			if(evaluations % (populationSize*10) == 0) {
				if(gd_.generationalDistance(previousPopulation_, population_, numObj_) <= 0.01*numObj_ && groupSize_ < numVar_/5) {
					groupSize_++;
					linearGrouping();
					if(evaluations % (populationSize*100) == 0) {
						System.out.println("The current group size is " + groupSize_);
					}
				} 
				savePreviousPopulation();
			}
			reproduction();
			environmentalSelection();	
			evaluations += populationSize;
		}
		NondominatedRanking final_ranking = new NondominatedRanking(population_);
		return final_ranking.getSubfront(0);
	}
	
	public void savePreviousPopulation() {
		previousPopulation_.clear();
		for(int i=0;i<populationSize;i++) {
			previousPopulation_.add(population_.get(i));
		}
	}
	
	public void initialization() throws JMException, ClassNotFoundException{
		evaluations = 0;
		populationSize = ((Integer) getInputParameter("populationSize")).intValue();
		maxEvaluations = ((Integer) getInputParameter("maxEvaluations")).intValue();
		population_ = new SolutionSet(populationSize);
		previousPopulation_ = new SolutionSet(populationSize);
		offspringPopulation_ = new SolutionSet(populationSize);
		unionPopulation_ = new SolutionSet(2*populationSize);
		//Read the operators
		mutationOperator_ = operators_.get("mutation");
		crossoverOperator_ = operators_.get("crossover");
		//Create the initial population
		Solution newSolution;
		for (int i = 0; i < populationSize; i++) {
			newSolution = new Solution(problem_);
			problem_.evaluate(newSolution);
			problem_.evaluateConstraints(newSolution);
			population_.add(newSolution);
		} // for
		zideal_ = new double[problem_.getNumberOfObjectives()];
		nadir_ = new double[problem_.getNumberOfObjectives()];
		UB_ = new double[numVar_];
		LB_ = new double[numVar_];
		for (int i = 0; i < numVar_; i++) {
			UB_[i] = problem_.getUpperLimit(i);
			LB_[i] = problem_.getLowerLimit(i);
		} // for
		groupSize_ = 5;
		linearGrouping();
		gd_ = new GenerationalDistance();
		savePreviousPopulation();
	}
	
	public void reproduction() throws JMException{
		double learningRate;
		Solution[] parents = new Solution[3];
		for (int i = 0; i < population_.size(); i++) {
			// obtain parents
			matingSelection(parents,i);
			XReal[] xParents = new XReal[3];
			for(int p=0; p<3; p++) {
				xParents[p] = new XReal(parents[p]);
			}
			Solution child = new Solution(parents[0]);
			learningRate = PseudoRandom.randDouble(0.0, 1.0);
			//encode
			double[][] encodedParents = encode(xParents);
            //search in the compressed space
			double[] newEncode = searchInTransferedSpace(encodedParents,learningRate);
			//decode
			decode(newEncode,child);
			//evaluation of the child
			mutationOperator_.execute(child);
			problem_.evaluate(child);
			offspringPopulation_.add(child);
		}
	}//for reproduction
	
	public void matingSelection(Solution[] parents,int i) {
		parents[0] = population_.get(i);
		int rdInt1 = PseudoRandom.randInt(0, populationSize-1);
		while(rdInt1 == i) {
			rdInt1 = PseudoRandom.randInt(0, populationSize-1);
		}
		int rdInt2 = PseudoRandom.randInt(0, populationSize-1);
		while(rdInt2 == i || rdInt2 == rdInt1) {
			rdInt2 = PseudoRandom.randInt(0, populationSize-1);
		}
		parents[1] = population_.get(rdInt1);
		parents[2] = population_.get(rdInt2);
	}
	
	public double[][] encode(XReal[] parents) throws JMException {
		int len = parents.length;
		double[][] encodedParents = new double[len][groupSize_];
		for(int l=0;l<len;l++) {
			double[] encodedSolution = new double[numVar_];
			encodedSolution = normalizedVariable(parents[l]);
			for(int g=0;g<groupSize_;g++) {
				encodedParents[l][g] = 0;
				for(int var=0;var<group_[g].length;var++) {
					encodedParents[l][g] += encodedSolution[group_[g][var]];
				}
				encodedParents[l][g] = encodedParents[l][g]/group_[g].length;
			}
		}
		return encodedParents; 
	}
	
	public double[] searchInTransferedSpace(double[][] encodedParents, double learningRate) {
		double[] newEncode = new double[groupSize_];
		for(int j=0;j<groupSize_;j++) {
			newEncode[j] = encodedParents[0][j] + learningRate*(encodedParents[1][j] - encodedParents[2][j]);
			if(newEncode[j] < 0) { newEncode[j] = 0.00001; } 
			if(newEncode[j] > 1) { newEncode[j] = 0.99999; }
		}
		return newEncode;
	}
	
	public void decode(double[] newEncode, Solution child) throws JMException {
		XReal offspring = new XReal(child);
		for(int g=0;g<groupSize_;g++) {
			double sum = 0;
			for(int var=0;var<group_[g].length;var++) {
				int cVar = group_[g][var];
				sum += (offspring.getValue(cVar)-LB_[cVar])/(UB_[cVar] - LB_[cVar]);
			}
			if(sum == 0) sum = 0.00001;
			for(int var=0;var<group_[g].length;var++) {
				int cVar = group_[g][var];
				double normalizedValue = (offspring.getValue(cVar)-LB_[cVar])/(UB_[cVar] - LB_[cVar]);
				double decodedValue = (normalizedValue/sum) * newEncode[g] * group_[g].length;
				decodedValue = decodedValue*(UB_[cVar] - LB_[cVar]) + LB_[cVar];
				if(decodedValue < LB_[cVar]) {
					decodedValue = LB_[cVar] + 0.00001;
				}
				if(decodedValue > UB_[cVar]) {
					decodedValue = UB_[cVar] - 0.00001;
				}
				offspring.setValue(cVar, decodedValue);
			}
		}
	}
	
	public double[] normalizedVariable(XReal solution) throws JMException {
		double[] x = new double[numVar_];
		for (int i = 0; i < numVar_; i++) 
			x[i] = solution.getValue(i);
		for (int i = 0; i < numVar_; i++) {
			x[i] = (x[i] - LB_[i])  / (UB_[i] - LB_[i]);
		}
		return x;
	}
	
	public void randomGrouping(){
		int gSize_ = numVar_/groupSize_;
		group_ = new int[groupSize_][];
		for(int g=0;g<groupSize_-1;g++){
			group_[g] = new int[gSize_];
		}
		int lSize_ = numVar_-(groupSize_-1)*gSize_;//the variable size of the last group
		group_[groupSize_-1] = new int[lSize_];
		int[] permutation = new int[numVar_];
		Utils.randomPermutation(permutation, numVar_);
		int t = 0;
		for(int g=0;g<groupSize_-1;g++){
			for(int m=0;m<gSize_;m++){
				group_[g][m] = permutation[t];
				t++;
			}
		}
		//assign variable to the last group
		for(int m=0;m<lSize_;m++){
			group_[groupSize_-1][m] = permutation[t];
			t++;
		}
	}
	
	public void linearGrouping(){
		int gSize_ = numVar_/groupSize_;
		group_ = new int[groupSize_][];
		for(int g=0;g<groupSize_-1;g++){
			group_[g] = new int[gSize_];
		}
		int lSize_ = numVar_-(groupSize_-1)*gSize_;//the variable size of the last group
		group_[groupSize_-1] = new int[lSize_];
		
		int t = 0;
		for(int g=0;g<groupSize_-1;g++){
			for(int m=0;m<gSize_;m++){
				group_[g][m] = t;
				t++;
			}
		}
		//assign variable to the last group
		for(int m=0;m<lSize_;m++){
			group_[groupSize_-1][m] = t;
			t++;
		}
	}
	/*
	 * Estimate the Ideal Point 
	 */
	public void estimateIdealPoint(SolutionSet solutionSet){
		for(int i=0; i<numObj_;i++){
			zideal_[i] = 1.0e+30;
			for(int j=0; j<solutionSet.size();j++){
				if(solutionSet.get(j).getObjective(i) < zideal_[i]){
					zideal_[i] = solutionSet.get(j).getObjective(i);
				}//if
			}//for
		}//for
	}
	
	/*
	 * Estimate the Ideal Point 
	 */
	public void estimateNadirPoint(SolutionSet solutionSet){
		for(int i=0; i<numObj_;i++){
			nadir_[i] = -1.0e+30;
			for(int j=0; j<solutionSet.size();j++){
				if(solutionSet.get(j).getObjective(i) > nadir_[i]){
					nadir_[i] = solutionSet.get(j).getObjective(i);
				}//if
			}//for
		}//for
	}
	
	public void normalization(SolutionSet solSet){
		double value;
		for(int p=0;p<solSet.size();p++){
			double sum = 0.0;
			
			for(int i=0;i<numObj_;i++){
				value = (solSet.get(p).getObjective(i)-zideal_[i])/(nadir_[i]-zideal_[i]);
				//value = (solSet.get(p).getObjective(i)-zIdeal[i]);
				solSet.get(p).setNormalizedObjective(i, value);
				sum += value;
			}
			solSet.get(p).setSumValue(sum);
			
			for(int i=0;i<numObj_;i++){
				solSet.get(p).setIthTranslatedObjective(i, solSet.get(p).getNormalizedObjective(i)/sum);
			}
		}
	}
	
	public void bestSolutionSelection(List<SolutionSet> list){
    	
    	for(int k=0; k<problem_.getNumberOfObjectives();k++){
    		double minClustering2Axis = Math.acos(Math.abs(list.get(0).getCentroid().getNormalizedObjective(k)
    						/list.get(0).getCentroid().getDistanceToIdealPoint()));
      		int minClustering2AxisID = 0;
      		for(int i=1;i<list.size();i++){
      			SolutionSet sols = list.get(i);
      			if(sols.size() == 0){
      				System.out.println("ElitSolutionSelection_Diversity_SolsSize = "+sols.size());
      				System.exit(0);
      			}
      			
      			double angle1 = Math.acos(Math.abs(sols.getCentroid().getNormalizedObjective(k)/sols.getCentroid().getDistanceToIdealPoint()));
      			//System.out.println(angle1);
      			if(angle1 < minClustering2Axis){
      				minClustering2Axis = angle1;
      				minClustering2AxisID = i;
      			}//if
      		}//for
      		double minSolution2Axis = Math.acos(list.get(minClustering2AxisID).get(0).getNormalizedObjective(k)
      				/list.get(minClustering2AxisID).get(0).getDistanceToIdealPoint());;
      		int minSolution2AxisID = 0;
      		for(int j=1;j<list.get(minClustering2AxisID).size();j++){
      			Solution sol = list.get(minClustering2AxisID).get(j);
      			double ang = Math.acos(list.get(minClustering2AxisID).get(j).getNormalizedObjective(k)
      					/list.get(minClustering2AxisID).get(j).getDistanceToIdealPoint());
      			if(ang < minSolution2Axis){
      				minSolution2Axis = ang;
      				minSolution2AxisID = j;
      			}
      		}//for
      		double rnd = PseudoRandom.randDouble();
      		if(rnd < 0.9){
      			population_.add(list.get(minClustering2AxisID).get(minSolution2AxisID));
              	list.remove(minClustering2AxisID);
      		}
    	}
    	
    	
	   Iterator<SolutionSet> it = list.iterator();
		while(it.hasNext()){
			SolutionSet sols = it.next();
			if(sols.size() == 0){
				System.out.println("In best solution selection, SolsSize2 = "+sols.size());
				System.exit(0);
			}
			if(sols.size() == 0){
				System.out.println("size = 0!");
				System.exit(0);
			}
			double minFitness = sols.get(0).getSumValue();
			int minFitnessID = 0;
			for(int j=1;j<sols.size();j++){
				Solution sol2 = sols.get(j);
				double fitness = sol2.getSumValue();
				if(minFitness > fitness){
					minFitness = fitness;
					minFitnessID = j;
				}	
			}//for
			population_.add(sols.get(minFitnessID));
			it.remove();
		}//while
		if(list.size() != 0){
			System.out.println("In best solution selection, ListSize2 = "+list.size());
			System.exit(0);
		}
   }
	
	public void environmentalSelection(){
		// Create the solutionSet union of solutionSet and offSpring
		unionPopulation_ = ((SolutionSet) population_).union(offspringPopulation_);
		population_.clear();
		offspringPopulation_.clear();
		SolutionSet st = getStSolutionSet(unionPopulation_,(int)(1.1*populationSize));
		estimateIdealPoint(st);
		estimateNadirPoint(st);
		normalization(st);
		List<SolutionSet> list = new <SolutionSet>ArrayList();
		for(int i=0;i<st.size();i++){
			SolutionSet sols = new SolutionSet();
			sols.add(st.get(i));
		    list.add(sols);  
		}
		list = new HierarchicalClustering(list).clusteringAnalysis(populationSize);
		if(list.size() != populationSize){
			System.out.println("The number of clusters after hierarchical clustering: ListSize = "+list.size());
			System.exit(0);
		}
		bestSolutionSelection(list);
	}

	public SolutionSet getStSolutionSet(SolutionSet ss,int size) {
		Ranking ranking = new NondominatedRanking(ss);
		int remain = size;
		int index = 0;
		SolutionSet front = null;
		SolutionSet mgPopulation = new SolutionSet();
		front = ranking.getSubfront(index);
		while ((remain > 0) && (remain >= front.size())) {

			for (int k = 0; k < front.size(); k++) {
				mgPopulation.add(front.get(k));
			} // for
			// Decrement remain
			remain = remain - front.size();
			// Obtain the next front
			index++;
			if (remain > 0) {
				front = ranking.getSubfront(index);
			} // if
		}
		if (remain > 0) { // front contains individuals to insert
			for (int k = 0; k < front.size(); k++) {
				mgPopulation.add(front.get(k));
			}
		}
		return mgPopulation;
	}
}