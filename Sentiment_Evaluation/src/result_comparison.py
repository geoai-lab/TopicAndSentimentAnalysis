
import csv
import matplotlib.pyplot as plt
from pylab import xticks
from matplotlib.ticker import FormatStrFormatter
import matplotlib



def read_ground_truth():
    ground_truth_data = {}
    
    csvfile = open('../data/Annals_Final_Test/Annal_paper_truth.csv', 'rb') 
    csvreader = csv.DictReader(csvfile)
    
    for row in csvreader:
        
        reviewID = row['reviewID']
        result_object = {"communityFriendliness":float(row['communityFriendliness']),"crimeSafety":float(row['crimeSafety']),"cultureDiversity":float(row['cultureDiversity']),"employmentOpportunity":float(row['employmentOpportunity']),"housingCondition":float(row['housingCondition']),"lifeConvenience":float(row['lifeConvenience']),"localWeather":float(row['localWeather']),"transportationConvenience":float(row['transportationConvenience'])}
        
        ground_truth_data[reviewID] = result_object
        
    
    csvfile.close()
    return ground_truth_data



def read_lda_lara_result():
    
    data = {}
    
    csvfile = open('../data/Annals_Final_Test/LDA_and_LARA.csv', 'rb') 
    csvreader = csv.DictReader(csvfile)
    
    for row in csvreader:
        
        reviewID = row['Neighborhood'].replace(" ","_")+"|"+row['reviewer']
        result_object = {"communityFriendliness":float(row['communityFriendliness']),"crimeSafety":float(row['crimeSafety']),"cultureDiversity":float(row['cultureDiversity']),"employmentOpportunity":float(row['employmentOpportunity']),"housingCondition":float(row['housingCondition']),"lifeConvenience":float(row['lifeConvenience']),"localWeather":float(row['localWeather']),"transportationConvenience":float(row['transportationConvenience'])}
        
        data[reviewID] = result_object
        
    
    csvfile.close()
    return data



def read_naive_result():
    
    data = {}
    
    csvfile = open('../data/Annals_Final_Test/naive_sentiment.csv', 'rb') 
    csvreader = csv.DictReader(csvfile)
    
    for row in csvreader:
        
        reviewID = row['reviewID']
        result_object = {"communityFriendliness":float(row['communityFriendliness']),"crimeSafety":float(row['crimeSafety']),"cultureDiversity":float(row['cultureDiversity']),"employmentOpportunity":float(row['employmentOpportunity']),"housingCondition":float(row['housingCondition']),"lifeConvenience":float(row['lifeConvenience']),"localWeather":float(row['localWeather']),"transportationConvenience":float(row['transportationConvenience'])}
        
        data[reviewID] = result_object
        
    
    csvfile.close()
    return data



def read_senti_word_result():
    
    data = {}
    
    csvfile = open('../data/Annals_Final_Test/word_sentiment.csv', 'rb') 
    csvreader = csv.DictReader(csvfile)
    
    for row in csvreader:
        
        reviewID = row['reviewID']
        result_object = {"communityFriendliness":float(row['communityFriendliness']),"crimeSafety":float(row['crimeSafety']),"cultureDiversity":float(row['cultureDiversity']),"employmentOpportunity":float(row['employmentOpportunity']),"housingCondition":float(row['housingCondition']),"lifeConvenience":float(row['lifeConvenience']),"localWeather":float(row['localWeather']),"transportationConvenience":float(row['transportationConvenience'])}
        
        data[reviewID] = result_object
        
    
    csvfile.close()
    return data



# compare the output of one model with the human annotation result; the target_aspect parameter is for specifying the aspect for comparison
# To compare all aspects, use "All" for target_aspect
def compare(ground_truth_data, test_data, target_aspect):
    
    total_diff = 0.0
    count = 0.0
    
    tp = 0.0
    fp = 0.0
    fn = 0.0
    
    
    for reviewID in ground_truth_data:
        ground_scores = ground_truth_data[reviewID]
        test_scores = test_data[reviewID]
        
        
        # if we would like to compare all aspects
        if target_aspect == "All":
            
            for aspect in ground_scores:
                ground_aspect_score = ground_scores[aspect]
                test_aspect_score = test_scores[aspect]
                
                if (test_aspect_score == -1) and (ground_aspect_score == -1):
                    continue
                
                if (test_aspect_score == -1) and (ground_aspect_score != -1):
                    total_diff += 2.0
                    fn += 1.0
                elif (test_aspect_score != -1) and (ground_aspect_score == -1):
                    total_diff += 2.0
                    fp += 1.0
                else:
                    total_diff += abs(test_aspect_score - ground_aspect_score)
                    tp += 1.0
                
                count += 1.0
                
                
        # if we like to compare only one aspect            
        else:
            ground_aspect_score = ground_scores[target_aspect]
            test_aspect_score = test_scores[target_aspect]
            

            if (test_aspect_score == -1) and (ground_aspect_score == -1):
                    continue
                
            if (test_aspect_score == -1) and (ground_aspect_score != -1):
                total_diff += 2.0
                fn += 1.0
            elif (test_aspect_score != -1) and (ground_aspect_score == -1):
                total_diff += 2.0
                fp += 1.0
            else:
                total_diff += abs(test_aspect_score - ground_aspect_score)
                tp += 1.0
            
            count += 1.0
                             
            
    
    average_diff = total_diff / count
    
    print('ARL: '+str(average_diff))    
    
    #print('precision: '+str(tp/(tp+fp)))    
    #print('recall: '+str(tp/(tp+fn)))  
    
   


def draw_hist(ground_data, test_data, target_aspect, y_limit):
    
    ratings = []
    
    # note: we have only 1000 ground truth data record, and thus we will extract the corresponding 1000 data records from our test data
    for reviewID in ground_data:
        rating_scores = test_data[reviewID]
        
        # if we would like to get the scores of all aspects
        if target_aspect == "All":
            for this_aspect in rating_scores:
                this_aspect_score = rating_scores[this_aspect]
                if this_aspect_score != -1:
                    ratings.append(this_aspect_score)
        
        # if we only like to get the score of one specific aspect
        else:
            this_aspect_score = rating_scores[target_aspect]
            if this_aspect_score != -1:
                ratings.append(this_aspect_score)
        
    
    
    #print(len(ratings))
     
    # draw the histogram
    matplotlib.rc('xtick', labelsize=18) 
    matplotlib.rc('ytick', labelsize=18) 
    
    plt.hist(ratings,bins=4, range = (1,5), edgecolor='black', linewidth=1)  
    xticks([1.0,2.0,3.0,4.0,5.0])
    axes = plt.gca()
    axes.set_xlim([1.0,5.0])
    axes.set_ylim([0,y_limit])   # change the y limit based on your plots
    axes.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.show()           


if __name__ == '__main__':
    ground_data = read_ground_truth()
    test_data_naive = read_naive_result();# #
    test_data_word = read_senti_word_result()
    test_data_lara = read_lda_lara_result()
    
    
    # for all aspects
    # draw historgram
    #draw_hist(ground_data, ground_data,"All",800)
    #draw_hist(ground_data, test_data_naive,"All",800)
    #draw_hist(ground_data, test_data_word,"All",800)
    #draw_hist(ground_data, test_data_lara,"All",800)
#            
#            
#     #comparison
#    compare(ground_data, ground_data, "All")
    #compare(ground_data, test_data_naive, "All")
    #compare(ground_data, test_data_word, "All")
    #compare(ground_data, test_data_lara, "All")
    
    
    
    # for crime and safety
    # draw historgram
#     draw_hist(ground_data, ground_data,"crimeSafety", 200)
#     draw_hist(ground_data, test_data_naive,"crimeSafety", 200)
#     draw_hist(ground_data, test_data_word,"crimeSafety", 200)
#     draw_hist(ground_data, test_data_lara,"crimeSafety", 200)
# #        
# #        
# #     # comparison
#     compare(ground_data, test_data_naive, "crimeSafety")
#     compare(ground_data, test_data_word, "crimeSafety")
#     compare(ground_data, test_data_lara, "crimeSafety")
    
    
    
    # for communityFriendliness
    # draw historgram
#     draw_hist(ground_data, ground_data,"communityFriendliness", 200)
#     draw_hist(ground_data, test_data_naive,"communityFriendliness", 200)
#     draw_hist(ground_data, test_data_word,"communityFriendliness", 200)
#     draw_hist(ground_data, test_data_lara,"communityFriendliness", 200)
#         
#         
#     # comparison
#     compare(ground_data, test_data_naive, "communityFriendliness")
#     compare(ground_data, test_data_word, "communityFriendliness")
#     compare(ground_data, test_data_lara, "communityFriendliness")



    # for cultural diversity
    # draw historgram
#     draw_hist(ground_data, ground_data,"cultureDiversity", 200)
#     draw_hist(ground_data, test_data_naive,"cultureDiversity", 200)
#     draw_hist(ground_data, test_data_word,"cultureDiversity", 200)
#     draw_hist(ground_data, test_data_lara,"cultureDiversity", 200)
         
         
    #comparison
#     compare(ground_data, test_data_naive, "cultureDiversity")
#     compare(ground_data, test_data_word, "cultureDiversity")
#     compare(ground_data, test_data_lara, "cultureDiversity")
    
    
    
    # for localWeather
    # draw historgram
#     draw_hist(ground_data, ground_data,"localWeather", 200)
#     draw_hist(ground_data, test_data_naive,"localWeather", 200)
#     draw_hist(ground_data, test_data_word,"localWeather", 200)
#     draw_hist(ground_data, test_data_lara,"localWeather", 200)
#         
#         
#     # comparison
#     compare(ground_data, test_data_naive, "localWeather")
#     compare(ground_data, test_data_word, "localWeather")
#     compare(ground_data, test_data_lara, "localWeather")




    # for life convenience
    # draw historgram
    #draw_hist(ground_data, ground_data,"lifeConvenience", 200)
    #draw_hist(ground_data, test_data_naive,"lifeConvenience", 200)
    #draw_hist(ground_data, test_data_word,"lifeConvenience", 200)
    #draw_hist(ground_data, test_data_lara,"lifeConvenience", 200)
         
         
    #comparison
#     compare(ground_data, test_data_naive, "lifeConvenience")
#     compare(ground_data, test_data_word, "lifeConvenience")
#     compare(ground_data, test_data_lara, "lifeConvenience")
    
    
    
    
    
    
    # for employmentOpportunity
    # draw historgram
#     draw_hist(ground_data, ground_data,"employmentOpportunity", 200)
#     draw_hist(ground_data, test_data_naive,"employmentOpportunity", 200)
#     draw_hist(ground_data, test_data_word,"employmentOpportunity", 200)
#     draw_hist(ground_data, test_data_lara,"employmentOpportunity", 200)
#          
#          
#     # comparison
#     compare(ground_data, test_data_naive, "employmentOpportunity")
#     compare(ground_data, test_data_word, "employmentOpportunity")
#     compare(ground_data, test_data_lara, "employmentOpportunity")
    
    
    
    # for transportation convenience
    # draw historgram
    draw_hist(ground_data, ground_data,"transportationConvenience", 200)
    draw_hist(ground_data, test_data_naive,"transportationConvenience", 200)
    draw_hist(ground_data, test_data_word,"transportationConvenience", 200)
    draw_hist(ground_data, test_data_lara,"transportationConvenience", 200)
          
          
    # comparison
    compare(ground_data, test_data_naive, "transportationConvenience")
    compare(ground_data, test_data_word, "transportationConvenience")
    compare(ground_data, test_data_lara, "transportationConvenience")
    
    
    
    
    
    # for transportation convenience
    # draw historgram
#     draw_hist(ground_data, ground_data,"housingCondition", 200)
#     draw_hist(ground_data, test_data_naive,"housingCondition", 200)
#     draw_hist(ground_data, test_data_word,"housingCondition", 200)
#     draw_hist(ground_data, test_data_lara,"housingCondition", 200)
#          
#          
#     # comparison
#     compare(ground_data, test_data_naive, "housingCondition")
#     compare(ground_data, test_data_word, "housingCondition")
#     compare(ground_data, test_data_lara, "housingCondition")
    
    
    
    