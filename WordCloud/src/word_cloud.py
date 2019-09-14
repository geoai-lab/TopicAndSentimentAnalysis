# This Python script visualizes the LDA results from R using word clouds


from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np


# a function for rendering the color of word clouds no need to change
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "rgb(10, 10, 10)"#"rgb(0, 0, 128)" #% random.randint(60, 100)



# Visualize the LDA result from the R code as word clouds
def visualize_R_LDA():
    k = 9    # 3, 7 , 8, 9
    topic_list = []
    for i in range(k):
        topic_list.append(dict())
    
    lineCount = 0
    with open('../data/LDAGibbs '+str(k)+' Topic_Term_Prob.csv','r') as fr:
        for line in fr:
            lineCount += 1
            if lineCount == 1:
                continue
            
            keyword_weight_array = line.split(',')
            for i in range(k):
                topic_list[i][keyword_weight_array[1+i*2].replace('"','')] = np.log2(float(keyword_weight_array[1+i*2+1])*1000)
            
            
     
    for i in range(k):    
        word_frequencies = topic_list[i]
        wordcloud = WordCloud(width=350, height=300, margin=8, prefer_horizontal=1, background_color='white',max_font_size=50).generate_from_frequencies(word_frequencies, max_font_size=None)
        plt.figure()
        plt.imshow(wordcloud.recolor(color_func= grey_color_func), interpolation='bilinear')
        plt.axis("off")
        plt.show()



if __name__ == '__main__':
    
    visualize_R_LDA()
    
    
    