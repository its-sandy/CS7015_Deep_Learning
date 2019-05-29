import os
import sys

flag = 1
sub  = sys.argv[1]
# print (sub)
comp = sub[-3:]

# print (comp)
# checking if the submission is in *.zip fromat
if comp != 'zip':
	print("ERROR: Please submit only in .zip format")
	exit()
	flag = 0

# Unzipping the submission
# print ('Unzipping the submission')
os.system('unzip {} -d submissions'.format(sub))

folder = os.listdir('submissions/')[0]
# print (folder)
files = os.listdir('submissions/{}'.format(folder))
# checking for Kaggle_subs folder
if 'Kaggle_subs' not in files:
	print ('ERROR: Please include Kaggle_subs folder')
	flag = 0
	files1 = []
else:
	files1 = os.listdir('submissions/{}/Kaggle_subs'.format(folder))

# print (files)
# print (files1)
inFiles  = ['report.pdf', 'run.sh', 'train.py']
csvFiles = ['secondbest.csv', 'firstbest.csv']

# files.remove('Kaggle_subs')
for f in inFiles:
	if f not in files:
		print ('ERROR: Please include {}'.format(f))
		flag = 0
		# exit()

for f in csvFiles:
	if f not in files1:
		print ('ERROR: Please include {} in Kaggle_subs folder'.format(f))
		flag = 0
		# exit()

if flag:
	print ('No errors, proceed to submit!!!')
os.system('rm -r submissions/')