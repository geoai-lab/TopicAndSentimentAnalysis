package aspectSegmenter;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class StopWordsRemover
{
		public static final String[] STOP_WORD_VALUES = new String[] {"crown","staten","manhattan","brooklyn","astoria","harlem","queen","nyc","york","bronx", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "de", "down", "due", "during", "each", "eg", "either","else", "elsewhere", "etc", "ever", "every", "everything", "everywhere", "except", "few", "for", "former", "formerly", "from", "front", "further", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "herself", "have","him", "himself", "his", "how", "however", "ie", "if", "in", "inc", "indeed", "into", "is", "it", "its", "itself", "least", "ltd", "may", "me", "meanwhile", "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "neither", "never", "nevertheless", "next", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "perhaps", "please", "put", "rather", "re", "see", "seem", "seemed", "seeming", "seems", "several", "she", "should", "show", "since", "sincere", "so", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "toward", "towards", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"};
		public static final Set<String> STOP_WORD_SET = new HashSet<String>(Arrays.asList(STOP_WORD_VALUES));
		
		public static String removeStopWords(String originalString)
		{
				originalString = originalString.toLowerCase();
				String[] wordsInOriginString = originalString.split(" ");
				StringBuffer resultStringBuffer = new StringBuffer();
				
				int size = wordsInOriginString.length;
				
				for (int i = 0; i < size; i++)
				{
						if(wordsInOriginString[i].length() <= 2) continue;
						
						if(STOP_WORD_SET.contains(wordsInOriginString[i])) continue;
						
						resultStringBuffer.append(wordsInOriginString[i]+" ");
				}
				if(resultStringBuffer.length()>0)
						return resultStringBuffer.substring(0, resultStringBuffer.length()-1);
				else {
						return "";
				}
		}

}
