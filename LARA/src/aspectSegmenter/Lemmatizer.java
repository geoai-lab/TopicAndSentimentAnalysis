package aspectSegmenter;

import java.util.Properties;

import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

public class Lemmatizer 
{
	static Lemmatizer m_lemmatizer;
	Properties props;
	StanfordCoreNLP pipeline;

	private Lemmatizer() {
		props = new Properties();
		props.put("annotators", "tokenize,ssplit, pos,  lemma");
		
		pipeline = new StanfordCoreNLP(props, false);
	}

	public static Lemmatizer getInstance() {
		if (m_lemmatizer == null) {
			m_lemmatizer = new Lemmatizer();
		}
		return m_lemmatizer;

	}

	public String getLemma(String text) {
		String lemma = "";
		Annotation document = pipeline.process(text);
		for (CoreMap sentence : document.get(SentencesAnnotation.class)) {
			for (CoreLabel token : sentence.get(TokensAnnotation.class)) {
				lemma += token.get(LemmaAnnotation.class) + " ";
			}
		}
		lemma = lemma.trim();
		return lemma;
	}

	public static void main(String[] args) {
		System.out.println(Lemmatizer.getInstance().getLemma("caribbean"));
	}
}