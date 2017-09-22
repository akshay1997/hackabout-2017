from __future__ import division
from itertools import izip

dictio={0:'Other',1:'Message-Topic', 2:'Message-Topic',
                 3:'Product-Producer', 4:'Product-Producer',
                 5:'Instrument-Agency', 6:'Instrument-Agency',
                 7:'Entity-Destination', 8:'Entity-Destination',
                 9:'Cause-Effect', 10:'Cause-Effect',
                 11:'Component-Whole', 12:'Component-Whole', 
                 13:'Entity-Origin', 14:'Entity-Origin',
                 15:'Member-Collection', 16:'Member-Collection',
                 17:'Content-Container', 18:'Content-Container'}

def opfile(mylist):
    i=8001
    with open('output.txt', 'w') as f:
        for data in mylist:
            f.write(str(i)+'\t'+dictio[data]+'\n')
            i+=1

def get_accuracy(file1,file2):
    c=0
    m=0
    with open(file1) as textfile1, open(file2) as textfile2: 
        for x, y in izip(textfile1, textfile2):
            x = x.strip()
            y = y.strip()
            c+=1
            if x==y:
                m+=1
    accuracy=(m/c)*100
    print('The accuracy is: ',accuracy)
    print('The number of matched is: ',m)
    print('Total number is: ',c)

if __name__ == '__main__':
    opfile(extract('output_akshay.txt'))
    get_accuracy('output.txt','TEST_FILE_KEY.TXT')
   
