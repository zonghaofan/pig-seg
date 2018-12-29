import os
import csv

def create_csv(dirname):
    path = './data/'+ dirname +'/'
    name = os.listdir(path)
    #name.sort(key=lambda x: int(x.split('.')[0]))
    #print(name)
    count = 0
    with open('data_'+dirname+'.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for n in name:
            if n[-4:] == '.jpg':
                # print(n)
                count += 1
                # with open('data_'+dirname+'.csv','rb') as f:
                writer.writerow(['./data/'+str(dirname) +'/'+ str(n), './data/' + str(dirname) + '_lbl/' + str(n)])
            else:
                pass
    print('{} {} examples.'.format(count, dirname))

if __name__ == "__main__":
    create_csv('trn')
    create_csv('val')

