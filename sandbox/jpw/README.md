# sandbox development environment

this is a folder to experiment with nltk

## env setup
1. unpacke source data to data folder
	* (platform dependant, 7z as samller than zip and meets github 100MB max file size limit)
	1. ../data/Train_rev1.7z
	1. ../data/Test_rev1.7z
	1. ../data/Validate_rev1.7z
1. install conda
	1. https://conda.io/docs/user-guide/install/index.html
1. conda update conda
1. conda create -n py35 python=3.5 anaconda
1. source activate py35
1. pip install -r requirments.txt
1. python
	1. import nltk
	1. nltk.download('stopwords')


