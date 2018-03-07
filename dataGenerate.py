import os


def Data_Gen(type,filename,classname):
	#TRAINING DATA
	#filename = 'data/train/cats'
	LIST = os.listdir(filename)
	for list in LIST:
		#print(list)
		msg = filename + '/' + list + ' ' + classname;
		f = open('data/'+type+'_data.txt','a')
		f.write(msg + '\n')
		print('Writting done FOR => ' + msg)

def main():
	type = 'train'
	filename = 'data/train/dogs' 
	classname =  '0'
	
	Data_Gen(type,filename,classname)
	
if __name__== "__main__":
  main()


	

