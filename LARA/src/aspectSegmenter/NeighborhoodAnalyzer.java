package aspectSegmenter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map;
import java.util.Vector;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.Span;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.StdOut;
import aspectSegmenter.NeighborhoodReview.Sentence;
import aspectSegmenter.NeighborhoodReview.Token;

public class NeighborhoodAnalyzer{
	
	// The aspects identified through topic modeling
	public static final String[] ASPECT_SET_NEW = {"Housing Conditions", "Transportation Convenience", "Employment Opportunity", "Life Convenience", "Local Weather", "Culture Diversity", "Community Friendliness", "Crime & Safety" };
	
	public static final int ASPECT_COUNT_CUT = 2;
	public static final int ASPECT_CONTENT_CUT = 5;
	public static final String PUNCT = ":;=+-()[],.\"'";
	
	class _Aspect{
		String m_name;
		HashSet<String> m_keywords;
		
		_Aspect(String name, HashSet<String> keywords){
			m_name = name;
			m_keywords = keywords;
		}
	}
	
	Vector<Neighborhood> m_neighborhoodList;	
	Vector<_Aspect> m_keywords;
	Hashtable<String, Integer> m_vocabulary;//indexed vocabulary
	Vector<String> m_wordlist;//term list in the original order
	
	HashSet<String> m_stopwords;
	Vector<rank_item<String>> m_ranklist;
	double[][] m_chi_table;
	double[] m_wordCount;
	boolean m_isLoadCV; // if the vocabulary is fixed
	
	//specific parameter to be tuned for bootstrapping aspect segmentation
	static public double chi_ratio = 4.0;
	static public int chi_size = 35;
	static public int chi_iter = 10;
	static public int tf_cut = 10;
	
	//NLP modules
	SentenceDetectorME m_stnDector;
	TokenizerME m_tokenizer;
	POSTaggerME m_postagger;
	Stemmer m_stemmer;	
		
	class rank_item<E> implements Comparable<rank_item<E>>{
		E m_name;
		double m_value;
		
		public rank_item(E name, double value){
			m_name = name;
			m_value = value;
		}
		
		@Override
		public int compareTo(rank_item<E> v) {
			if (m_value < v.m_value) return 1;
			else if (m_value > v.m_value) return -1;
			return 0;
		}
		
	}
	
	public NeighborhoodAnalyzer(String seedwords, String stopwords, String stnSplModel, String tknModel, String posModel){
		m_neighborhoodList = new Vector<Neighborhood>();
		m_vocabulary = new Hashtable<String, Integer>();
		m_chi_table = null;
		m_isLoadCV = false;
		if (seedwords != null && seedwords.isEmpty()==false)
			LoadKeywords(seedwords);
		LoadStopwords(stopwords);
		
		try {
			m_stnDector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnSplModel)));
			m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tknModel)));
			m_postagger = new POSTaggerME(new POSModel(new FileInputStream(posModel)));
			m_stemmer = new Stemmer();
		} catch (InvalidFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("[Info]NLP modules initialized...");
	}
	
	public void LoadKeywords(String filename){
		try {
			m_keywords = new Vector<_Aspect>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
			String tmpTxt;
			String[] container;
			HashSet<String> keywords;
			while( (tmpTxt=reader.readLine()) != null ){
				container = tmpTxt.split("\t");
				keywords = new HashSet<String>(container.length-1);
				for(int i=1; i<container.length; i++)
					keywords.add(container[i]);
				m_keywords.add(new _Aspect(container[0], keywords));
				System.out.println("Keywords for " + container[0] + ": " + keywords.size());
			}
			reader.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void LoadVocabulary(String filename){
		try {
			m_vocabulary = new Hashtable<String, Integer>();
			m_wordlist = new Vector<String>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			String[] container;
			while( (tmpTxt=reader.readLine()) != null ){
				container = tmpTxt.split("\t");
				m_vocabulary.put(container[0], m_vocabulary.size());
				m_wordlist.add(tmpTxt.trim());
			}
			reader.close();
			m_isLoadCV = true;
			System.out.println("[Info]Load " + m_vocabulary.size() + " control terms...");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void LoadStopwords(String filename){
		try {
			m_stopwords = new HashSet<String>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			while( (tmpTxt=reader.readLine()) != null )
				m_stopwords.add(tmpTxt.toLowerCase());
			reader.close();
			
			m_stopwords = (HashSet<String>)StopWordsRemover.STOP_WORD_SET;
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
		
	public String[] getLemma(String[] tokens){
		String[] lemma = new String[tokens.length];
		String term;
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
	
	
	private String cleanReview(String content){
		String error_A = "showReview\\([\\d]+\\, [\\w]+\\);";//
		return content.replace(error_A, "");
	}
	
	//the review format is fixed
	public void LoadReviews(String filename){//load reviews for annotation purpose
		try {
			String neighborhoodID = filename.substring(filename.lastIndexOf("/")+1).replace(".csv", "").trim();   // neighborhood ID and neighborhood name are the same
			Neighborhood tNeighborhood = new Neighborhood(neighborhoodID);
			tNeighborhood.m_name = neighborhoodID;
			
			CSVParser csvParser = new CSVParser(new FileReader(filename),CSVFormat.EXCEL.withHeader());
			for(CSVRecord csvRecord: csvParser)
			{
				// first read data from the csv file
				String userIDString = csvRecord.get("userID");
				String reviewString = csvRecord.get("review");
				//String createdTimeString = csvRecord.get("createdTime");
				String overallRatingString = csvRecord.get("rating");
				
				// second, construct review objects
				NeighborhoodReview review = null;
				String[] stns, tokens;
				Span[] stn_spans = null;
				
				String content = cleanReview(reviewString);
				if(content != null)
					stn_spans = m_stnDector.sentPosDetect(content); //list of the sentence spans
				/*if (stn_spans.length<2){
					continue;
				}*/
				
				stns = Span.spansToStrings(stn_spans, content);
				String reviewID = neighborhoodID+"_"+userIDString;
				review = new NeighborhoodReview(neighborhoodID, reviewID, Integer.valueOf(overallRatingString),content);
				for(int i=0; i<stns.length; i++){
					//stns[i] = stns[i].replaceAll("[^a-zA-Z]"," ").replaceAll("\\s+"," ").toLowerCase().trim();
					//stns[i] = StopWordsRemover.removeStopWords(stns[i]);
					tokens = m_tokenizer.tokenize(stns[i]); //stns[i].split(" ");//
					if (tokens!=null && tokens.length>0)//discard too short sentences
						review.addStn(stns[i], tokens, m_postagger.tag(tokens), getLemma(tokens), m_stopwords);
			    }
				
				//if (review.getStnSize()>= 2)    // if a review has more than 2 sentences, add it to the neighborhood desc; note, the initial is > 2
			//	{
					if (m_isLoadCV==false) //didn't load the controlled vocabulary
						expendVocabular(review);
					tNeighborhood.addReview(review);
				//}
				

			}
			csvParser.close();
				
			if (tNeighborhood.getReviewSize()>1){
				m_neighborhoodList.add(tNeighborhood);
				if (m_neighborhoodList.size()%100==0)
					System.out.print(".");
				if (m_neighborhoodList.size()%10000==0)
					System.out.println(".");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void LoadDirectory(String path, String suffix){		
		File dir = new File(path);
		int size = m_neighborhoodList.size();
		File[] files = dir.listFiles();
		Arrays.sort(files);
		for (File f : files) {
			if (f.isFile() && f.getName().endsWith(suffix))
				LoadReviews(f.getAbsolutePath());
			else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		size = m_neighborhoodList.size() - size;
		
		System.out.println("Loading " + size + " neighborhoods from " + path);
		
		for(int i=0;i<m_neighborhoodList.size();i++)
		{
			System.out.println(m_neighborhoodList.get(i).toPrintString());
		}
	}
	
	//save for hReviews
	public void Save2Vectors(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "US-ASCII"));
			int[][] vectors = new int[m_keywords.size()][m_vocabulary.size()];
			double[] ratings = new double[1+m_keywords.size()], counts = new double[1+m_keywords.size()];
			int aspectID, wordID, outputSize=0, reviewSize=0;
			for(Neighborhood neighborhood:m_neighborhoodList){
				for(NeighborhoodReview r:neighborhood.m_reviews){//aggregate all the reviews
					Annotate(r);
					
					reviewSize++;
					for(aspectID=0; aspectID<ratings.length; aspectID++){
						//if (r.m_ratings[aspectID]>0){
							ratings[aspectID] += r.m_overall_rating; //r.m_ratings[aspectID];
							counts[aspectID] += 1;//only take the average in the existing ratings
						//}
					}
					
					//collect the vectors
					for(Sentence stn:r.m_stns){
						if ((aspectID = stn.m_aspectID)<0)
							continue;

						for(Token t:stn.m_tokens){//select the in-vocabulary word
							if (m_vocabulary.containsKey(t.m_lemma)){
								wordID = m_vocabulary.get(t.m_lemma);
								vectors[aspectID][wordID]++;
							}
						}
					}
					
					Save2Vector(writer, r.m_reviewID, reviewSize, ratings, counts, vectors);
					outputSize ++;
					clearVector(ratings, counts, vectors);
					
				}						
				
			}
			
			writer.close();
			//System.out.println("Output " + outputSize + " neighborhood reviews...");
			System.out.println("Finish converting neighborhood reviews to vectors ...");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void Save2Vector(BufferedWriter writer, String hotelID, int reviewSize, double[] ratings, double[] counts, int[][] vectors) throws IOException{
		int aspectID, wordID;
		DecimalFormat formater = new DecimalFormat("#.###");
		writer.write(hotelID);
		double score;
		for(aspectID=0; aspectID<ratings.length; aspectID++){
			if (counts[aspectID]>0)
				score = ratings[aspectID]/counts[aspectID];
			else 
				score = ratings[0]/counts[0];//using overall rating as default
			writer.write("\t" + formater.format(score));
		}
		writer.write("\n");
		
		for(aspectID=0; aspectID<vectors.length; aspectID++){
			for(wordID=0; wordID<vectors[aspectID].length; wordID++){
				if (vectors[aspectID][wordID]>0)
					writer.write(wordID + ":" + vectors[aspectID][wordID] + " ");
			}
			writer.write("\n");
		}
	}
	
	private void clearVector(double[] ratings, double[] counts, int[][] vectors){
		Arrays.fill(ratings, 0);
		Arrays.fill(counts, 0);
		for(int aspectID=0; aspectID<vectors.length; aspectID++)
			Arrays.fill(vectors[aspectID], 0);
	}
	
	private void expendVocabular(NeighborhoodReview tReview){
		for(Sentence stn : tReview.m_stns){
			for(Token t : stn.m_tokens){
				if (m_vocabulary.containsKey(t.m_lemma) == false)
					m_vocabulary.put(t.m_lemma, m_vocabulary.size());
			}
		}
	}
	
	private void createChiTable(){
		if (m_chi_table==null){
			m_chi_table = new double[m_keywords.size()][m_vocabulary.size()];
			m_wordCount = new double[m_vocabulary.size()];
			
			m_ranklist = new Vector<rank_item<String>>(m_vocabulary.size());
			Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();
			while(vIt.hasNext()){
				Map.Entry<String, Integer> entry = vIt.next();
				m_ranklist.add(new rank_item<String>(entry.getKey(), 0.0));
			}
		}
		else{
			for(int i=0; i<m_chi_table.length; i++)
				Arrays.fill(m_chi_table[i], 0.0);
			Arrays.fill(m_wordCount, 0.0);
		}
	}
	
	private void Annotate(NeighborhoodReview tReview){
		for(Sentence stn : tReview.m_stns){
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
	
	private void collectStats(NeighborhoodReview tReview){
		int aspectID, wordID;
		for(Sentence stn : tReview.m_stns){
			if ( (aspectID=stn.getAspectID())>-1){
				for(Token t:stn.m_tokens){
					if (m_vocabulary.containsKey(t.m_lemma) == false)
						System.out.println("Missing:" + t);
					else{
						wordID = m_vocabulary.get(t.m_lemma);
						m_chi_table[aspectID][wordID] ++;
					}
				}
			}
		}
	}
	
	/**
	 * 
	 * @param A: w and c
	 * @param B: w and !c
	 * @param C: !w and c
	 * @param D: !w and !c
	 * @param N: total
	 * @return Chi-Sqaure
	 */
	private double ChiSquareValue(double A, double B, double C, double D, double N){
		double denomiator = (A+C) * (B+D) * (A+B) * (C+D);
		if (denomiator>0 && A+B > tf_cut)
			return N * (A*D-B*C) * (A*D-B*C) / denomiator;
		else
			return 0.0;//problematic setting (this word hasn't been assigned)
	}
	
	private void ChiSquareTest(){		
		createChiTable();
		for(Neighborhood neighborhood:m_neighborhoodList){
			for(NeighborhoodReview tReview:neighborhood.m_reviews){
				Annotate(tReview);
				collectStats(tReview);
			}
		}
		
		double N = 0;
		double[] aspectCount = new double[m_keywords.size()];
		int i, j;
		for(i=0; i<aspectCount.length; i++){
			for(j=0; j<m_wordCount.length; j++){
				aspectCount[i] += m_chi_table[i][j];
				m_wordCount[j] += m_chi_table[i][j];
				N += m_chi_table[i][j];
			}
		}
		
		for(i=0; i<aspectCount.length; i++){
			for(j=0; j<m_wordCount.length; j++){
				m_chi_table[i][j] = ChiSquareValue(m_chi_table[i][j], 
													m_wordCount[j]-m_chi_table[i][j], 
													aspectCount[i]-m_chi_table[i][j],
													N-m_chi_table[i][j], N);
			}
		}
	}
	
	private void getVocabularyStat(){
		if (m_wordCount==null)
			m_wordCount = new double[m_vocabulary.size()];
		else
			Arrays.fill(m_wordCount, 0);
		
		for(Neighborhood neighborhood:m_neighborhoodList){
			for(NeighborhoodReview tReview:neighborhood.m_reviews){
				for(Sentence stn : tReview.m_stns){					
					for(Token t:stn.m_tokens){
						if (m_vocabulary.containsKey(t.m_lemma) == false){
							System.out.println("Missing:" + t);
						} else{
							int wordID = m_vocabulary.get(t.m_lemma);
							m_wordCount[wordID] ++;
						}
					}
				}
			}
		}
	}
	
	private boolean expandKeywordsByChi(double ratio){
		int wordID, aspectID, selID = -1;
		double maxChi, chiV;
		Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();
		while(vIt.hasNext()){//first iteration, select the maxAspect
			Map.Entry<String, Integer> entry = vIt.next();
			wordID = entry.getValue();
			
			maxChi = 0.0;
			selID = -1;
			for(aspectID=0; aspectID<m_keywords.size(); aspectID++){
				if ((chiV=m_chi_table[aspectID][wordID]) > ratio * maxChi){
					maxChi = chiV;
					selID = aspectID;
				}
			}
			
			for(aspectID=0; aspectID<m_keywords.size(); aspectID++){
				if (aspectID!=selID)
					m_chi_table[aspectID][wordID] = 0.0;
			}
		}
		
		aspectID = 0;
		boolean extended = false;
		for(int i=0; i<m_keywords.size(); i++){	
			_Aspect asp = m_keywords.get(i);
			for(rank_item<String> item : m_ranklist){//second iteration, select the maxAspect
				wordID = m_vocabulary.get(item.m_name);
				item.m_value = m_chi_table[aspectID][wordID];	
			}
			
			Collections.sort(m_ranklist);
			for(wordID=0; wordID<chi_size; wordID++){
				if (asp.m_keywords.add(m_ranklist.get(wordID).m_name))
					extended = true;
			}
			aspectID ++;
		}
		return extended;
	}
	
	public void OutputChiTable(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			int wordID, aspectID;
			for(aspectID=0; aspectID<m_keywords.size(); aspectID++)
				writer.write("\t" + m_keywords.get(aspectID).m_name);
			writer.write("\n");
			
			Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();			
			while(vIt.hasNext()){
				Map.Entry<String, Integer> entry = vIt.next();
				wordID = entry.getValue();
				writer.write(entry.getKey());
				for(aspectID=0; aspectID<m_keywords.size(); aspectID++)
					writer.write("\t" + m_chi_table[aspectID][wordID]);
				writer.write("\n");
			}
			writer.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void SaveVocabulary(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			getVocabularyStat();
			
			Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();
			while(vIt.hasNext()){//iterate over all the words
				Map.Entry<String, Integer> entry = vIt.next();
				int wordID = entry.getValue();
				
				writer.write(entry.getKey() + "\t" + m_wordCount[wordID] + "\n");
			}
			writer.close();
			
			System.out.println("[Info]Vocabulary size: " + m_vocabulary.size());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//output word list with more statistic info: CHI, DF 
	public void OutputWordListWithInfo(String filename){
		System.out.println("Vocabulary size: " + m_vocabulary.size());
		try {
			ChiSquareTest();//calculate the chi table
			
			int wordID, aspectID;
			double chi_value; 
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();
			while(vIt.hasNext()){//iterate over all the words
				Map.Entry<String, Integer> entry = vIt.next();
				wordID = entry.getValue();
				
				chi_value = 0;
				for(aspectID=0; aspectID<m_keywords.size(); aspectID++)
					chi_value += m_chi_table[aspectID][wordID];//calculate the average Chi2
				
				if (chi_value/aspectID>3.84 && m_wordCount[wordID]>50)
					writer.write(entry.getKey() + "\t" + (chi_value/aspectID) + "\t" + m_wordCount[wordID] + "\n");
			}
			writer.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void BootStrapping(String filename){
		System.out.println("Vocabulary size: " + m_vocabulary.size());
		
		int iter = 0;
		do {
			ChiSquareTest();
			System.out.println("Bootstrapping for " + iter + " iterations...");
		}while(expandKeywordsByChi(chi_ratio) && ++iter<chi_iter );
		
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			for(int i=0; i<m_keywords.size(); i++){
				_Aspect asp = m_keywords.get(i);
				writer.write(asp.m_name);
				Iterator<String> wIter = asp.m_keywords.iterator();
				while(wIter.hasNext())
					writer.write(" " + wIter.next());
				writer.write("\n");
			}
			writer.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void readSeedFile(String filename)
	{
		try 
		{
			m_keywords = new Vector<>();
			In inputFile = new In(filename);
			String inputline = null;
			//int index = 0;
			while ((inputline = inputFile.readLine())!= null) 
			{
				String[] keywordsInfo = inputline.split(" ");
				HashSet<String> keywords = new HashSet<>();
				for(int i=1;i<keywordsInfo.length;i++)
					keywords.add(keywordsInfo[i].trim());
				
				_Aspect thisAspect = new _Aspect(keywordsInfo[0].trim(), keywords);
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
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
	
	
	// the main function
	static public void main(String[] args){
		NeighborhoodAnalyzer analyzer = new NeighborhoodAnalyzer("Data/Seeds/neighborhood_bootstrapping_init.dat", "Data/Seeds/stopwords.dat", 
				"Data/Model/NLP/en-sent.zip", "Data/Model/NLP/en-token.zip", "Data/Model/NLP/en-pos-maxent.bin");
		
		// load neighborhood review data
		analyzer.LoadDirectory("Data/NeighborhoodReviews/", ".csv");
		
		// expanding the aspect keywords using bootstrapping
		//analyzer.BootStrapping("Data/Seeds/neighborhood_bootstrapping_expanded.dat");
		
		// using an expanded and revised seed file
		analyzer.readSeedFile("Data/Seeds/neighborhood_bootstrapping_May10.dat");
		
		// convert each review into vector representation (which will be used in the next step for decomposing overall ratings)
		analyzer.Save2Vectors("Data/Vectors/vector_neighborhood_4000.dat");	
	}
}