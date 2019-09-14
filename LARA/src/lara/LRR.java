package lara;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Vector;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import aspectSegmenter.Lemmatizer;
import aspectSegmenter.NeighborhoodReview;
import aspectSegmenter.NeighborhoodReview.Sentence;
import aspectSegmenter.StopWordsRemover;
import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.Out;
import edu.princeton.cs.algs4.StdOut;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;
import optimizer.LBFGS;
import optimizer.LBFGS.ExceptionWithIflag;
import utilities.Utilities;

public class LRR extends RatingRegression {
	static public boolean SIGMA = false;
	static public double PI = 0.5;
	static final public int K = 8; // A default value if the system cannot detect the number of aspects; 

	public LRR_Model m_model; 
	protected double[] m_old_alpha; // in case optimization for alpha failed
	BufferedWriter m_trace;
	
	// aspect will be determined by the input file for LRR
	public LRR(int alphaStep, double alphaTol, int betaStep, double betaTol, double lambda){
		super(alphaStep, alphaTol, betaStep, betaTol, lambda);
		
		m_model = null;
		m_old_alpha = null;
	}
	
	// if we want to load previous models
	public LRR(int alphaStep, double alphaTol, int betaStep, double betaTol, double lambda, String modelfile){
		super(alphaStep, alphaTol, betaStep, betaTol, lambda);
		
		m_model = new LRR_Model(modelfile);
		m_old_alpha = new double[m_model.m_k];
	}
	
	@Override
	protected double init(int v){
		super.init(v);
		double initV = 1;// likelihood for the first iteration won't matter
		
		//keep track of the model update trace 
		try {
			m_trace = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("trace.dat"), "UTF-8"));
			for(int i=0; i<m_k; i++)
				m_trace.write(String.format("Aspect_%d\t" , i));
			m_trace.write("alpha\tbeta\tdata\taux_data\tsigma\n");//column title for the trace file
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		if (m_model==null){
			m_model = new LRR_Model(m_k, v);
			m_old_alpha = new double[m_model.m_k];
	
			PI = 2.0;//try to seek a better initialization of beta
			initV = MStep(false);//this is just estimated alpha, no need to update Sigma yet
			PI = 0.5;
		}
		
		return initV;
	}
	
	@Override
	protected double prediction(Vector4Review vct){
		//Step 1: infer the aspect ratings/weights
		EStep(vct);
		
		//Step 2: calculate the overall rating
		double orating = 0;
		for(int i=0; i<m_model.m_k; i++)
			orating += vct.m_alpha[i] * vct.m_pred[i];
		return orating;
	}
	
	protected double EStep(Vector4Review vct){
		//step 1: estimate aspect rating
		vct.getAspectRating(m_model.m_beta);
		
		//step 2: infer aspect weight
		try {
			System.arraycopy(vct.m_alpha, 0, m_old_alpha, 0, m_old_alpha.length);
			return infer_alpha(vct);
		} catch (ExceptionWithIflag e) {
			System.arraycopy(m_old_alpha, 0, vct.m_alpha, 0, m_old_alpha.length);//failed with exceptions
			return -2;
		}
	}
	
	//we are estimating \hat{alpha}
	protected double getAlphaObjGradient(Vector4Review vct){
		double expsum = Utilities.expSum(vct.m_alpha_hat), orating = -vct.m_ratings[0], s, sum = 0;
		
		// initialize the gradient
		Arrays.fill(m_g_alpha, 0);
		
		for(int i=0; i<m_model.m_k; i++){
			vct.m_alpha[i] = Math.exp(vct.m_alpha_hat[i])/expsum;//map to aspect weight
			
			orating += vct.m_alpha[i] * vct.m_pred[i];//estimate the overall rating
			m_alpha_cache[i] = vct.m_alpha_hat[i] - m_model.m_mu[i];//difference with prior
			
			s = PI*(vct.m_pred[i]-vct.m_ratings[0]) * (vct.m_pred[i]-vct.m_ratings[0]);
			
			if (Math.abs(s)>1e-10){//in case we will disable it
				for(int j=0; j<m_model.m_k; j++){
					if (j==i)
						m_g_alpha[j] += 0.5 * s * vct.m_alpha[i]*(1-vct.m_alpha[i]); 
					else
						m_g_alpha[j] -= 0.5 * s * vct.m_alpha[i]*vct.m_alpha[j];
				}
				sum += vct.m_alpha[i] * s;		
			}
		}
		
		double diff = orating/m_model.m_delta;
		for(int i=0; i<m_model.m_k; i++){
			s = 0;
			for(int j=0; j<m_model.m_k; j++){
				// part I of objective function: data likelihood
				if (i==j)
					m_g_alpha[j] += diff*vct.m_pred[i] * vct.m_alpha[i]*(1-vct.m_alpha[i]);
				else
					m_g_alpha[j] -= diff*vct.m_pred[i] * vct.m_alpha[i]*vct.m_alpha[j];
				
				// part II of objective function: prior
				s += m_alpha_cache[j] * m_model.m_sigma_inv[i][j];
			}
			
			m_g_alpha[i] += s;
			sum += m_alpha_cache[i] * s;
		}		
		
		return 0.5 * (orating*orating/m_model.m_delta + sum);
	}
	
	protected double infer_alpha(Vector4Review vct) throws ExceptionWithIflag{
		double f = 0;
		int iprint [] = {-1,0}, iflag[] = {0}, icall = 0, n = m_model.m_k, m = 5;

		//initialize the diagonal matrix
		Arrays.fill(m_diag_alpha, 0);
		do {
			f = getAlphaObjGradient(vct);//to be minimized
			LBFGS.lbfgs ( n , m , vct.m_alpha_hat , f , m_g_alpha , false , m_diag_alpha , iprint , m_alphaTol , 1e-20 , iflag );
		} while ( iflag[0] != 0 && ++icall <= m_alphaStep );
		
		if (iflag[0]!=0)
			return -1; // have not converged yet
		else{
			double expsum = Utilities.expSum(vct.m_alpha_hat);
			for(n=0; n<m_model.m_k; n++)
				vct.m_alpha[n] = Math.exp(vct.m_alpha_hat[n])/expsum;
			return f;
		}
	}
	
	private void testAlphaVariance(boolean updateSigma){
		try {
			int i;
			double v;
			
			//test the variance of \hat\alpha estimation
			Arrays.fill(m_diag_alpha, 0.0);
			for(Vector4Review vct:m_collection){
				if (vct.m_4train==false)
					continue; // do not touch testing cases
				
				for(i=0; i<m_k; i++){
					v = vct.m_alpha_hat[i] - m_model.m_mu[i];
					m_diag_alpha[i] += v * v; // just for variance
				}
			}
			
			for(i=0; i<m_k; i++){
				m_diag_alpha[i] /= m_trainSize;
				if (i==0 && updateSigma)
					m_trace.write("*");
				m_trace.write(String.format("%.3f:%.3f\t", m_model.m_mu[i], m_diag_alpha[i]));//mean and variance of \hat\alpha
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//m-step can only be applied to training samples!!
	public double MStep(boolean updateSigma){
		updateSigma = false; // shall we update Sigma?
		int i, j, k = m_model.m_k;
		double v;
		
		//Step 0: initialize the statistics
		Arrays.fill(m_g_alpha, 0.0);
		
		//Step 1: ML for \mu
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			for(i=0; i<k; i++)
				m_g_alpha[i] += vct.m_alpha_hat[i];
		}
		for(i=0; i<k; i++)
			m_model.m_mu[i] = m_g_alpha[i]/m_trainSize;
		testAlphaVariance(updateSigma);
		
		
		//Step 2: ML for \sigma
		if (updateSigma){//we may choose to not update \Sigma
			//clear up the cache
			for(i=0; i<k; i++)
				Arrays.fill(m_model.m_sigma_inv[i], 0);
			
			for(Vector4Review vct:m_collection){
				if (vct.m_4train==false)
					continue; // do not touch the testing cases
				
				for(i=0; i<k; i++)
					m_diag_alpha[i] = vct.m_alpha_hat[i] - m_model.m_mu[i];
				
				if(SIGMA){//estimate the whole covariance matrix
					for(i=0; i<k; i++){
						for(j=0; j<k; j++){
							m_model.m_sigma_inv[i][j] += m_diag_alpha[i] * m_diag_alpha[j];
						}
					}
				} else {// just for estimate diagonal
					for(i=0; i<k; i++)
						m_model.m_sigma_inv[i][i] += m_diag_alpha[i] * m_diag_alpha[i]; 
				}
			}
			
			for(i=0; i<k; i++){
				if (SIGMA){
					m_model.m_sigma_inv[i][i] = (1.0 + m_model.m_sigma_inv[i][i]) / (1 + m_trainSize); // prior
					for(j=0; j<k; j++)
						m_model.m_sigma.setQuick(i, j, m_model.m_sigma_inv[i][j]);
				} else {
					v = (1.0 + m_model.m_sigma_inv[i][i]) / (1 + m_trainSize);
					m_model.m_sigma.setQuick(i, i, v);
					m_model.m_sigma_inv[i][i] = 1.0 / v;
				}
			}
			m_model.calcSigmaInv(1);
		}
		
		//calculate the likelihood for the alpha part
		double alpha_likelihood = 0, beta_likelihood = 0;
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			for(i=0; i<k; i++)
				m_diag_alpha[i] = vct.m_alpha_hat[i] - m_model.m_mu[i];
			alpha_likelihood += m_model.calcCovariance(m_diag_alpha);
		}
		alpha_likelihood += Math.log(m_model.calcDet()); 
		
		//Step 3: ML for \beta
		try {
			ml_beta();
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		}
		
		beta_likelihood = getBetaPriorObj();
		
		//Step 4: ML for \delta
		double datalikelihood = getDataLikelihood(), auxdata = getAuxDataLikelihood(), oldDelta = m_model.m_delta;
		m_model.m_delta = datalikelihood / m_trainSize;	
		datalikelihood /= oldDelta;
		
		try {
			m_trace.write(String.format("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", alpha_likelihood, beta_likelihood, datalikelihood, auxdata, Math.log(m_model.m_delta)));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return alpha_likelihood + beta_likelihood + datalikelihood + auxdata + Math.log(m_model.m_delta);
	}
	
	//\beat^T * \beta
	protected double getBetaPriorObj(){
		double likelihood = 0;
		for(int i=0; i<m_model.m_beta.length; i++){
			for(int j=0; j<m_model.m_beta[i].length; j++)
				likelihood += m_model.m_beta[i][j] * m_model.m_beta[i][j];
		}
		return m_lambda * likelihood;
	}
	
	//\sum_d(\sum_i\alpha_{di}\S_{di}-r_d)^2/\sigma^2
	protected double getDataLikelihood(){
		double likelihood = 0, orating;
						
		// part I of objective function: data likelihood
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			orating = -vct.m_ratings[0];
			
			//apply the current model
			vct.getAspectRating(m_model.m_beta);
			for(int i=0; i<vct.m_alpha.length; i++)
				orating += vct.m_alpha[i] * vct.m_pred[i];
			likelihood += orating*orating;
		}
		return likelihood;
	}
	
	//\sum_d\pi\sum_i\alpha_{di}(\S_{di}-r_d)^2
	protected double getAuxDataLikelihood(){
		double likelihood = 0, orating;
						
		// part I of objective function: data likelihood
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			orating = vct.m_ratings[0];
			for(int i=0; i<vct.m_alpha.length; i++)
				likelihood += vct.m_alpha[i] * (vct.m_pred[i] - orating)*(vct.m_pred[i] - orating);
		}
		return PI * likelihood;
	}

	protected double getBetaObjGradient(){
		double likelihood = 0, aux_likelihood = 0, orating, diff, oRate;
		int vSize = m_model.m_v + 1, offset;
		SpaVector sVct;
		
		// initialize the structure
		Arrays.fill(m_g_beta, 0);
				
		// part I of objective function: data likelihood
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			oRate = vct.m_ratings[0];
			orating = -oRate;
			
			//apply the current model
			vct.getAspectRating(m_beta, vSize);
			for(int i=0; i<m_model.m_k; i++)
				orating += vct.m_alpha[i] * vct.m_pred[i];			
			
			likelihood += orating*orating;
			orating /= m_model.m_delta; // in order to get consistency between aux-likelihood
			
			offset = 0;
			for(int i=0; i<m_model.m_k; i++){
				aux_likelihood += vct.m_alpha[i]* (vct.m_pred[i]-oRate)*(vct.m_pred[i]-oRate);
				if (RatingRegression.SCORE_SQUARE)
					diff = vct.m_alpha[i]*(orating + PI*(vct.m_pred[i]-oRate)) * vct.m_pred_cache[i];
				else
					diff = vct.m_alpha[i]*(orating + PI*(vct.m_pred[i]-oRate)) * vct.m_pred[i];
				
				sVct = vct.m_aspectV[i];
				m_g_beta[offset] += diff;//first for bias term
				for(int j=0; j<sVct.m_index.length; j++)
					m_g_beta[offset + sVct.m_index[j]] += diff*sVct.m_value[j];
				offset += vSize;//move to next aspect
			}
		}
		
		double reg = 0;
		for(int i=0; i<m_beta.length; i++)
		{
			m_g_beta[i] += m_lambda*m_beta[i];
			reg += m_beta[i]*m_beta[i];
		}
		
		return 0.5*(likelihood/m_model.m_delta + PI*aux_likelihood + m_lambda*reg);
	}
	
	protected double ml_beta() throws ExceptionWithIflag{
		double f = 0;
		int iprint [] = {-1,0}, iflag[] = {0}, icall = 0, n = (1+m_model.m_v)*m_model.m_k, m = 5;
		
		for(int i=0; i<m_model.m_k; i++)//set up the starting point
			System.arraycopy(m_model.m_beta[i], 0, m_beta, i*(m_model.m_v+1), m_model.m_v+1);
		
		Arrays.fill(m_diag_beta, 0);
		do {
			if (icall%1000==0)
				System.out.print(".");//keep track of beta update
			f = getBetaObjGradient();//to be minimized
			LBFGS.lbfgs ( n , m , m_beta , f , m_g_beta , false , m_diag_beta , iprint , m_betaTol , 1e-20 , iflag );
		} while ( iflag[0] != 0 && ++icall <= m_betaStep );

		System.out.print(icall + "\t");
		for(int i=0; i<m_model.m_k; i++)
			System.arraycopy(m_beta, i*(m_model.m_v+1), m_model.m_beta[i], 0, m_model.m_v+1);
		return f;
	}
	
	public void EM_est(String filename, int maxIter, double converge){
		int iter = 0, alpha_exp = 0, alpha_cov = 0;
		double tag, diff = 10, likelihood = 0, old_likelihood = init(LoadVectors(filename));
				
		System.out.println("[Info]Step\toMSE\taMSE\taCorr\tiCorr\tcov(a)\texp(a)\tobj\tconverge");
		while(iter<Math.min(8, maxIter) || (iter<maxIter && diff>converge)){
			alpha_exp = 0;
			alpha_cov = 0;
			
			//E-step
			for(Vector4Review vct:m_collection){
				if (vct.m_4train){
					tag = EStep(vct);
					if (tag==-1) // failed to converge
						alpha_cov ++;
					else if (tag==-2) // failed with exceptions
						alpha_exp ++;
				}					
			}			
			System.out.print(iter + "\t");//sign of finishing E-step
			
			//M-step
			likelihood = MStep(iter%4==3);//updating \Sigma too often will hamper \hat\alpha convergence		
			
			evaluateAspect();// evaluating in the testing cases
			diff = (old_likelihood-likelihood)/old_likelihood;
			old_likelihood = likelihood;
			System.out.println(String.format("\t%d\t%d\t%.3f\t%.3f", alpha_cov, alpha_exp, likelihood, diff));
			iter++;
		}
		
		try {
			m_trace.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void SaveModel(String filename){
		m_model.Save2File(filename);
	}
	
	
	// Yingjie combine result code
	static void combineResult()
	{
		try 
		{
			In decomposedRatingFile = new In("Data/Results/prediction.dat");
			In reviewVectorFile = new In("Data/Vectors/vector_neighborhood_4000.dat");
			FileReader neighborhoodReader = new FileReader("Data/all_reviews.csv");
			CSVParser neighborhoodParser = new CSVParser(neighborhoodReader, CSVFormat.EXCEL.withHeader());
			Iterator<CSVRecord> neighborIterator = neighborhoodParser.iterator();
			
			// save the review contents into a hashtable
			Hashtable<String, String> reviewHashtable = new Hashtable<>();
			while(neighborIterator.hasNext())
			{
				CSVRecord thisNeigbhorReview = neighborIterator.next();
				String reviewID = thisNeigbhorReview.get("reviewID").trim();
				String reviewContent = thisNeigbhorReview.get("content");
				reviewHashtable.put(reviewID, reviewContent);
			}
			
			Out combinedResultFile = new Out("combinedResult.csv");
			combinedResultFile.println("Neighborhood,reviewer,rating,content,crimeSafety,housingCondition,transportationConvenience,employmentOpportunity,lifeConvenience,localWeather,cultureDiversity,communityFriendliness");
			
			Vector<_Aspect> m_keywords = readSeedFile("Data/Seeds/neighborhood_bootstrapping_May10.dat");
			SentenceDetectorME m_stnDector =new SentenceDetectorME(new SentenceModel(new FileInputStream("Data/Model/NLP/en-sent.zip")));
			TokenizerME m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream("Data/Model/NLP/en-token.zip")));
			POSTaggerME m_postagger = new POSTaggerME(new POSModel(new FileInputStream("Data/Model/NLP/en-pos-maxent.bin")));
			HashSet<String> m_stopwords= (HashSet<String>)StopWordsRemover.STOP_WORD_SET;//new HashSet<String>();
			
			String decomposedRatingLine = null;
			String reviewVectorLine = null;
			while((decomposedRatingLine = decomposedRatingFile.readLine()) != null)
			{
				String[] ratingResult = decomposedRatingLine.split("\t");
				String[] neighborhood_reviewerInfo = ratingResult[0].split("_");
				
				String overallRatingString = ratingResult[1];
				
				String reviewID = ratingResult[0].trim();
				String reviewContent = reviewHashtable.get(reviewID);
				if(reviewContent == null)
				{
					StdOut.println("Can't find review content for "+reviewID);
					continue;
				}
				
				reviewVectorLine = reviewVectorFile.readLine();
				boolean[] hasAspect = new boolean[m_keywords.size()];  // 8 topics
				for(int i=0;i<hasAspect.length;i++)
				{
					reviewVectorLine = reviewVectorFile.readLine().trim();
					if(reviewVectorLine.length() > 0)
						hasAspect[i] = true;
				}
				
				// begin normalization here
				// get the weight of each aspect
				String[] stns, tokens;
				Span[] stn_spans = null;
				
				String content = cleanReview(reviewContent);
				//if(content != null)
				stn_spans = m_stnDector.sentPosDetect(content); //list of the sentence spans
				stns = Span.spansToStrings(stn_spans, content);
				double overallRating = Double.valueOf(overallRatingString);
				NeighborhoodReview review = new NeighborhoodReview(neighborhood_reviewerInfo[0], reviewID, (int)overallRating,reviewContent);
				for(int i=0; i<stns.length; i++){
					//stns[i] = stns[i].replaceAll("[^a-zA-Z]"," ").replaceAll("\\s+"," ").toLowerCase().trim();
					//stns[i] = StopWordsRemover.removeStopWords(stns[i]);
					tokens = m_tokenizer.tokenize(stns[i]); //stns[i].split(" ");//
					if (tokens!=null && tokens.length>1)//discard too short sentences
						review.addStn(stns[i], tokens, m_postagger.tag(tokens), getLemma(tokens), m_stopwords);
			    }
				
				Annotate(review, m_keywords);
				int totalReviewLength = 0;
				double[] aspectLength = new double[m_keywords.size()];
				for(Sentence thisStns: review.m_stns)
				{
					int thisStnsLength = thisStns.getLength();
					int aspect = thisStns.getAspectID();
					if(aspect != -1)
					{
						aspectLength[aspect] += thisStnsLength;
						totalReviewLength += thisStnsLength;
					}
				}
				
				for(int i=0;i<aspectLength.length;i++)
					aspectLength[i] = aspectLength[i] / (totalReviewLength*1.0);
				
				
				//double smallestValue = 100;
				double[] score = new double[hasAspect.length];
				double normalSum = 0;
				int scoreCount = 0;
				for(int i=0;i<hasAspect.length;i++)
				{
					if(hasAspect[i])
					{
						score[i] = Double.valueOf(ratingResult[11+i].trim());
						normalSum += aspectLength[i] *score[i];
						//if(score[i]<smallestValue) smallestValue = score[i];
						scoreCount ++;
					}
				}
				
				//double weight = 1.0/scoreCount;
				
			/*	for(int i=0;i<hasAspect.length;i++)
				{
					if(hasAspect[i])
					{
						normalSum += weight*score[i];
					}
				}*/
				
				overallRating = Double.valueOf(overallRatingString);
				double normalPara = overallRating/normalSum;

				String outputResult = "\""+neighborhood_reviewerInfo[0]+"\","+neighborhood_reviewerInfo[1]+","+overallRatingString+",\""+reviewContent+"\"";
				
				for(int i=0;i<hasAspect.length;i++)
				{
					if(hasAspect[i])
					{
						double scaledScore = score[i]*normalPara;
						
						if(Double.isNaN(scaledScore) && (scoreCount == 1)) scaledScore = overallRating;
						//if(scoreCount == 1) scaledScore = overallRating;
						if(scaledScore>5) scaledScore = 5;
						if(scaledScore<1) scaledScore = 1;
						
						if(!Double.isNaN(scaledScore))
							outputResult += ","+String.format("%.3f", scaledScore);
						else
							outputResult += ",null";
					}
					else {
						outputResult += ",null";
					}
				}
				
				combinedResultFile.println(outputResult);
			}
			
			neighborhoodParser.close();
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
	
	private static String cleanReview(String content){
		String error_A = "showReview\\([\\d]+\\, [\\w]+\\);";//
		return content.replace(error_A, "");
	}
	
	private static void Annotate(NeighborhoodReview tReview, Vector<_Aspect> m_keywords){
		for(Sentence stn : tReview.m_stns){
			//if(tReview.m_content.equalsIgnoreCase("A lot of Asian and Eastern European cuisine"))
				//StdOut.println("test");
			int maxCount = 0, count, sel = -1;
			for(int index=0; index<m_keywords.size(); index++){				
				if ( (count=stn.AnnotateByKeyword(m_keywords.get(index).m_keywords))>maxCount ){
					maxCount = count;
					sel = index;
				}
				else if (count==maxCount)
					sel = -1;//don't allow tie
			}
			stn.setAspectID(sel);
		}
	}
	
	
	
	public static Vector<_Aspect> readSeedFile(String filename)
	{
		try 
		{
			Vector<_Aspect> m_keywords = new Vector<>();
			In inputFile = new In(filename);
			//int index= 0;
			String inputline = null;
			while ((inputline = inputFile.readLine())!= null) 
			{
				String[] keywordsInfo = inputline.split(" ");
				HashSet<String> keywords = new HashSet<>();
				for(int i=1;i<keywordsInfo.length;i++)
					keywords.add(keywordsInfo[i].trim());
				
				_Aspect thisAspect = new _Aspect(keywordsInfo[0].trim(), keywords);
				//m_keywords.add();
				m_keywords.add(thisAspect);
				//index++;
			}
			inputFile.close();
			
			// Test if the file is read successfullyfor(int i=0; i<m_keywords.size(); i++){
			for(int i=0;i<m_keywords.size();i++)
			{	
				_Aspect asp = m_keywords.get(i);
				StdOut.print(asp.m_name);
				Iterator<String> wIter = asp.m_keywords.iterator();
				while(wIter.hasNext())
					StdOut.print(" " + wIter.next());
				StdOut.println();
			}
			
			return m_keywords;
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
		
		return null;
	}
	
	public static String[] getLemma(String[] tokens){
		String[] lemma = new String[tokens.length];
		String term;
		String PUNCT = ":;=+-()[],.\"'";
		for(int i=0; i<lemma.length; i++){
			//lemma[i] = m_stemmer.stem(tokens[i].toLowerCase());//shall we stem it?
			term = tokens[i].toLowerCase();
			if (term.length()>1 && PUNCT.indexOf(term.charAt(0))!=-1 && term.charAt(1)>='a' && term.charAt(1)<='z')
				lemma[i] = term.substring(1);
			else 
				lemma[i] = term;
			
			lemma[i] = Lemmatizer.getInstance().getLemma(lemma[i]);
		}
		return lemma;
	}
	//  Combine result finished

	
	
	// the main function
	public static void main(String[] args) {
		// initialize the LRR model
		LRR model = new LRR(500, 1e-2, 5000, 1e-2, 2.0);
		
		// read the vector representations of neighborhood reviews
		model.EM_est("Data/Vectors/vector_neighborhood_4000.dat", 10, 1e-4);
		
		// save the prediction result
		model.SavePrediction("Data/Results/prediction.dat");
	    model.SaveModel("Data/Model/model_neighborhood.dat");
		
	    // combine the result into a single file
		LRR.combineResult();
	}
}




class _Aspect{
	String m_name;
	HashSet<String> m_keywords;
	
	_Aspect(String name, HashSet<String> keywords){
		m_name = name;
		m_keywords = keywords;
	}
}
