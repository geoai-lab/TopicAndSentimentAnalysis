package aspectSegmenter;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Vector;

public class Neighborhood implements Comparable<Neighborhood>, Serializable{
	
	private static final long serialVersionUID = -5695057513648909065L;
	public String m_ID;
	//public double m_rating;
	//public int m_price;
	//public String m_URL;
	public String m_name;
	
	//public String m_imageURL;
	//public String m_address;
	//public String m_price_range;
	
	public List<NeighborhoodReview> m_reviews;
	public double m_rScore;
	public int m_index;
	DecimalFormat m_formater;
	
	public Neighborhood(String ID, String name){
		m_ID = ID;
		m_name = name;
		//m_rating = rating;
		//m_price = price;
		//m_URL = URL;
		
		m_reviews = null;
		
		m_formater = new DecimalFormat("#.#");
	}
	
	public Neighborhood(String ID){
		m_ID = ID;
		m_name = null;
		//m_URL = null;
		//m_imageURL = null;
		//m_price = 0;
	//	m_price_range = null;
		
		m_reviews = new Vector<NeighborhoodReview>();
		m_formater = new DecimalFormat("#.#");
	}
	
	public int getReviewSize(){
		return m_reviews==null?0:m_reviews.size();
	}
	
	@Override
	public int compareTo(Neighborhood h) {
		if (this.m_rScore>h.m_rScore)
			return -1;
		else if (this.m_rScore<h.m_rScore)
			return 1;
		else 
			return 0;
	}
	
	public String toString(){
		return m_ID + "\n" + m_name;// + "\n" //+ m_rating; //+ "\n" ;//+ (m_price_range==null||m_price_range.isEmpty()?m_price:m_price_range) + "\n" + m_URL + "\n";
	}
	
	public String toPrintString(){
		StringBuffer buffer = new StringBuffer(512);
		buffer.append("<Neighborhood Name>"+m_name+"\n");
		//buffer.append("<Hotel Address>"+m_address+"\n");
		//buffer.append("<Overall Rating>"+m_rating+"\n");
		//buffer.append("<Avg. Price>"+m_price_range+"\n");
		//buffer.append("<URL>"+m_URL+"\n");
		//buffer.append("<Image URL>"+m_imageURL+"\n");		
		for(int i=0;i<m_reviews.size();i++)
			buffer.append(m_reviews.get(i).toPrintString());
		
		return buffer.toString();
	}		
		
	public void addReview(NeighborhoodReview r){
		if (m_reviews==null)
			m_reviews = new Vector<NeighborhoodReview>();
		m_reviews.add(r);
	}
}