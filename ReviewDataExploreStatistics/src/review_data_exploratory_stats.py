

import matplotlib.pyplot as plt



def review_length_hist():
    
    review_length = []
    with open('../data/NYC_Review_WordCount.csv','r') as fr:
        for line in fr:
            info = line.split(',')
            review_length.append(int(info[1]))
            
    plt.hist(review_length,bins=17,range = (0,340), edgecolor='black', linewidth=1)
    #plt.xlabel("Word Count")
    #plt.ylabel("Frequency")
    plt.show()
    



if __name__ == '__main__':
    review_length_hist()