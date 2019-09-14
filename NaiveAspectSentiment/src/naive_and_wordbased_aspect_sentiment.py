
import logging
from nltk.corpus import stopwords
import csv
import re

from gensim.models.wrappers.ldamallet import LdaMallet



def compute_sentiment(): 
    # load the trained LDA model
    lda = LdaMallet.load('../data/lda_8_topic_neighborhood_review_final.lda')
    lda_topic_object_array = lda.show_topics(num_topics=-1, num_words=20, log=False, formatted=False)      
    lda.print_topics(num_topics=-1, num_words=20)
    
    
    # load the sentiment dictionary, and convert the initial scale from -5 to 5 to 1 to 5
    sentiment_dict = {}
    with open('../data/AFINN-111.txt') as AFINN_file: 
        for line in AFINN_file:
            splitted = line.split('\t')
            word = splitted[0].strip()
            init_score = float(splitted[1].strip())
            new_score = (((init_score - (-5)) * (5 - 1)) / (5 - (-5))) + 1
            sentiment_dict[word] = new_score
            
      
    default_stopwords = stopwords.words('english')
    # avoid place names be considered as normal words
    additional_stopwords = ["manhattan","new york","nyc","brooklyn","bronx","queens"] 

    # file for storing the word-based sentiment analysis result
    wordbased_output_file = open('../data/word_sentiment.csv', 'w')
    
    # file for storing the naive sentiment analysis result
    naive_output_file = open('../data/naive_sentiment.csv', 'w')
    
    fieldnames = ['reviewID','neighborhood','reviewer','rating','content','crimeSafety','housingCondition','transportationConvenience','employmentOpportunity','lifeConvenience','localWeather','cultureDiversity','communityFriendliness']
    topic_mapping_dict = {'crimeSafety':5,'housingCondition':1,'transportationConvenience':2,'employmentOpportunity':7,'lifeConvenience':3,'localWeather':6,'cultureDiversity':0,'communityFriendliness':4}
    
    review_writer = csv.DictWriter(wordbased_output_file, fieldnames=fieldnames)
    review_writer.writeheader()
    
    naive_review_writer = csv.DictWriter(naive_output_file, fieldnames=fieldnames)
    naive_review_writer.writeheader()

    
    with open('../data/all_reviews.csv', 'rb') as csvfile:
        csvreader = csv.DictReader(csvfile)   
        for row in csvreader:
            #review_obj = {'reviewID':row['reviewID'],"neighborhood":row["neighborhood"], "authorID":row["authorID"],"overall_rating": float(row["overall_rating"]),"review_content":row["review_content"]}
            #print(review_obj)                
            #reviews.append(review_obj)
            #print("The review is: "+row["review_content"])
            
            neighborhood_name = row["neighborhood"].lower().strip()
            review_content = row["review_content"].lower()
            review_content = review_content.replace(neighborhood_name,'')
            
            for special_word in additional_stopwords:
                review_content = review_content.replace(special_word,'')
            review_content = re.sub('\s+', ' ', review_content)
            review_content = review_content.strip()
            
            # since we split sentences with comma, question mark, and others, we need to concatenate the short phrases
            review_sentence_array_raw = re.split('[;!?.,]', review_content)
            
            review_sentence_array = []
            for raw_sentence in review_sentence_array_raw:
                if len(raw_sentence.split()) <= 3:
                    if len(review_sentence_array) > 0:
                        review_sentence_array[-1] += " " + raw_sentence.strip()
                    else:
                        review_sentence_array.append(raw_sentence.strip())
                else:
                    review_sentence_array.append(raw_sentence)
                    
                    
            # create a dict object to store the sentiments of various aspects; -1 is for overall      
            review_senti_score_dict = {-1:{'count':0,'score':0}}
            for this_topic_object in lda_topic_object_array:
                review_senti_score_dict[this_topic_object[0]] = {'count':0,'score':0}
            
            
            for sentence in review_sentence_array:
                #print('The sentence is: ' + sentence)
                sentence = re.sub('[^a-zA-Z]', ' ', sentence)
                sentence = re.sub('\s+', ' ', sentence)
                sentence = sentence.strip()
                if len(sentence) == 0:
                    continue
                sentence_words = [word for word in sentence.split() if word not in default_stopwords and len(word) > 1]
                
                # judge which topic this sentence is about
                sentence_topic_dict = {}
                for this_topic_object in lda_topic_object_array:
                    sentence_topic_dict[this_topic_object[0]] = 0
                    for this_keyword_object in this_topic_object[1]:
                        for this_sentence_word in sentence_words:
                            if this_keyword_object[0] == this_sentence_word:
                                sentence_topic_dict[this_topic_object[0]] += this_keyword_object[1]
      
                sentence_final_topic = -1
                maxi_topic_value = 0
                for topic in sentence_topic_dict:
                    if (sentence_topic_dict[topic] >= maxi_topic_value) and (sentence_topic_dict[topic]!= 0):
                        
                        if sentence_topic_dict[topic] == maxi_topic_value:
                            print("find tie "+ str(sentence_topic_dict[topic]))
                            
                        sentence_final_topic = topic
                        maxi_topic_value = sentence_topic_dict[topic]
                
                
                
                # add the sentiment score for each sentence
                for this_sentence_word in sentence_words:
                    if(sentiment_dict.has_key(this_sentence_word)):
                        if sentence_final_topic != -1:
                            review_senti_score_dict[sentence_final_topic]['score'] += sentiment_dict[this_sentence_word]
                            review_senti_score_dict[sentence_final_topic]['count'] += 1
                            
                        review_senti_score_dict[-1]['score'] += sentiment_dict[this_sentence_word]
                        review_senti_score_dict[-1]['count'] += 1
                
            
            
            review_senti_result = ''
            naive_review_senti_result = ''
            overallRating = float(row["overall_rating"])
            
            naive_review_senti_score_dict = {}  # this is for the naive approach (where all the aspects are the same)
            
            for topic in review_senti_score_dict:
                count = review_senti_score_dict[topic]['count']
                score = review_senti_score_dict[topic]['score']
                
                if count > 0:
                    avg_score = float(score)/float(count)
                    review_senti_score_dict[topic]['score'] = avg_score
                    review_senti_result += " topic:"+str(topic)+", score:"+str(avg_score)+"; "
                    
                    naive_review_senti_score_dict[topic] = overallRating
                    naive_review_senti_result += " topic:"+str(topic)+", score:"+str(overallRating)+"; "
                    
                else:
                    review_senti_score_dict[topic]['score'] = -1
                    naive_review_senti_score_dict[topic] = -1
            
            print(review_senti_result.strip())
            
            
            print(str(review_senti_score_dict))
            
            
            review_writer.writerow({'reviewID':row['reviewID'],'neighborhood':row["neighborhood"],'reviewer':row["authorID"],'rating':float(row["overall_rating"]),'content':row["review_content"],'crimeSafety':review_senti_score_dict[topic_mapping_dict['crimeSafety']]['score'],'housingCondition':review_senti_score_dict[topic_mapping_dict['housingCondition']]['score'],'transportationConvenience':review_senti_score_dict[topic_mapping_dict['transportationConvenience']]['score'],'employmentOpportunity':review_senti_score_dict[topic_mapping_dict['employmentOpportunity']]['score'],'lifeConvenience':review_senti_score_dict[topic_mapping_dict['lifeConvenience']]['score'],'localWeather':review_senti_score_dict[topic_mapping_dict['localWeather']]['score'],'cultureDiversity':review_senti_score_dict[topic_mapping_dict['cultureDiversity']]['score'],'communityFriendliness':review_senti_score_dict[topic_mapping_dict['communityFriendliness']]['score']})
            naive_review_writer.writerow({'reviewID':row['reviewID'],'neighborhood':row["neighborhood"],'reviewer':row["authorID"],'rating':float(row["overall_rating"]),'content':row["review_content"],'crimeSafety':naive_review_senti_score_dict[topic_mapping_dict['crimeSafety']],'housingCondition':naive_review_senti_score_dict[topic_mapping_dict['housingCondition']],'transportationConvenience':naive_review_senti_score_dict[topic_mapping_dict['transportationConvenience']],'employmentOpportunity':naive_review_senti_score_dict[topic_mapping_dict['employmentOpportunity']],'lifeConvenience':naive_review_senti_score_dict[topic_mapping_dict['lifeConvenience']],'localWeather':naive_review_senti_score_dict[topic_mapping_dict['localWeather']],'cultureDiversity':naive_review_senti_score_dict[topic_mapping_dict['cultureDiversity']],'communityFriendliness':naive_review_senti_score_dict[topic_mapping_dict['communityFriendliness']]})
              
                
    wordbased_output_file.close() 
    naive_output_file.close()          
    
    


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    compute_sentiment()
    
