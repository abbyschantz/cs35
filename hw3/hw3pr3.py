#
#
# Names: Liz Harder, Abby Schantz, Eliana Keinan
# 
# File: hw3pr3.py


"""
The data set that we contains the findings based on a large-scale survey of fake news. 
More specifically, the data shows how a specific user reacts to 11 potential articles. Each article 
has been officailly labeled as True or False (Which we will called is_fake). In the accuracy_bool, 
the participant states if they think the article is True or False (since this is the preceived 
accuracy, we will call this prec_true). Because of the way the data is given, the state of the article 
is given as FALSE if the article is Real and TRUE if the article is fake. This is the opposite of the 
way that the preceived accuracy is presented as it is FALSE if the article is preceived to be false and 
TRUE if the article is preceived to be real. 

In datavis1, we use the information on accuracy to plot out how many articles are False and 
preceived false, True and preceived true, False and precieved true, and True and precieved false. 

In datavis2, we 

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import numpy as np
import pylab
from matplotlib.patches import Polygon
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from csv import reader


with open('headline-responses.csv', 'r') as f:
    data = list(reader(f))

is_fake = [i[8]for i in data]
prec_true = [i[6]for i in data]
headline = [i[1]for i in data]
row_count = sum(1 for row in data)

def news_match():
    """ This function returns the number of submissions where the article was true/false 
    news and it was also preceived to be true/false by the viwer. There are four categories 
    all written in the formatt of (article state_preceived state)"""

    real_real = 0 #real news and preceived to be real
    real_fake = 0 #real news and preceived to be fake
    fake_real = 0 #fake news and preceived to be real
    fake_fake = 0 #fake news and preceived to be fake

    for i in range(row_count):
        if is_fake[i] == 'FALSE' and prec_true[i] == 'TRUE':
            real_real += 1
        if is_fake[i] == 'FALSE' and prec_true[i] == 'FALSE':
            real_fake += 1
        if is_fake[i] == 'TRUE' and prec_true[i] == 'TRUE':
            fake_real += 1
        if is_fake[i] == 'TRUE' and prec_true[i] == 'FALSE':
            fake_fake += 1
 
    return real_real, real_fake, fake_real, fake_fake

#The following formulas calculate the percentage of each of the 4 categories using row count as the total number of entries
rr_perc = (news_match()[0]/row_count)*100
rf_perc = (news_match()[1]/row_count)*100
fr_perc = (news_match()[2]/row_count)*100
ff_perc = (news_match()[3]/row_count)*100


def datavis1():
    """This data visualization function inputs the formulas from above that calculate 
    the percentage of responses in each of the four categories and outputs a pie chart 
    of these results
    """
    with plt.xkcd():
    # The slices will be ordered and plotted counter-clockwise.
        labels = 'Real News is Real', 'Real News is Fake', 'Fake News is Real', 'Fake News is Fake'
        sizes = [rr_perc, rf_perc, fr_perc, ff_perc]
        colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
        explode = (0, 0, 0.2, 0) # only "explode" the 2nd slice (i.e. 'Hogs')

        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        plt.annotate('Percentage of people who think that...', xy=(0,1), xycoords='axes fraction' )
        # Set aspect ratio to be equal so that pie is drawn as a circle.
        plt.axis('equal')
        plt.show()

def article_count():
    """this function inputs the headline-responses.csv column regarding which 
    article (A,B,C,D,E,F,G,H,I,J,K) was recognized and counts the total number. 
    The function returns the recognition count for each article
    """
    art_a = 0
    art_b = 0
    art_c = 0
    art_d = 0
    art_e = 0
    art_f = 0
    art_g = 0
    art_h = 0
    art_i = 0
    art_j = 0
    art_k = 0

    for i in range(row_count):
        if headline[i] == 'A':
            art_a += 1
        if headline[i] == 'B':
            art_b += 1
        if headline[i] == 'C':
            art_c += 1
        if headline[i] == 'D':
            art_d += 1
        if headline[i] == 'E':
            art_e += 1
        if headline[i] == 'F':
            art_f += 1
        if headline[i] == 'G':
            art_g += 1
        if headline[i] == 'H':
            art_h += 1
        if headline[i] == 'I':
            art_i += 1
        if headline[i] == 'J':
            art_j += 1
        if headline[i] == 'K':
            art_k += 1

    return art_a, art_b, art_c, art_d, art_e, art_f, art_g, art_h, art_i, art_j, art_k

def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def datavis2():
    """This data visualization function inputs the formulas from above 
    that calculate the amount of times each article was recalled by participants 
    and outputs a bar graph of an article recognition score
    The articles are as follows:
    A: Pope Francis Shocks World, Endorses Donald Trump for President, Releases Statement (Fake)
    B: Donald Trump Sent His Own Plane to Transport 200 Stranded Marines (Fake)
    C: FBI Agent Suspected in Hillary Email Leaks Found Dead in Apparent Murder - Suicide (Fake)
    D: Donald Trump Protester Speaks Out: “I Was Paid $3,500 to Protest Trump’s Rally” (Fake)
    E: FBI Director Comey Just Put a Trump Sign On His Front Lawn (Fake)
    F: Melania Trump’s Girl-on-Girl Photos From Racy Shoot Revealed (True)
    G: Barbara Bush: “I don’t know how women can vote” for Trump (True)
    H: Donald Trump Says He’d ‘Absolutely’ Require Muslims to Register (True)
    I: Trump: “I Will Protect Our LGBTQ Citizens” (True)
    J: I Ran the C.I.A Now I’m Endorsing Hillary Clinton (True)
    K: Donald Trump on Refusing Presidential Salary: “I’m not taking it” (True)
    """
    n_groups = 11
    scores = (article_count()[0], article_count()[1] ,article_count()[2] ,article_count()[3],article_count()[4],article_count()[5],article_count()[6],article_count()[7],article_count()[8],article_count()[9],article_count()[10])
    fig, ax = plt.subplots()
    

    index = np.arange(n_groups)
    bar_width = 0.5

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, scores, bar_width,
                    alpha=opacity,
                    color='b',
                    error_kw=error_config)

    plt.xlabel('Article')
    plt.ylabel('Recognition Population')
    plt.title('Recognition Population of Each Article')
    plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E','F','G','H','I','J','K'))
    ax.text(1977, 67,
        ("Percentage of the US Population\n"
         "carrying cameras everywhere they go,\n"
         "every waking moment of their lives:"),
        size=16)

    plt.tight_layout()
    plt.show()
