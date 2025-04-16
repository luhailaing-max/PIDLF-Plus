
First you must run Data_prepare.py to prepare the data
Environment：
(1)Main python package:
	pytorch 1.9
	python 3.9.18
	kapok
	gdal 3.0.2
	.....

(2)The kapok pakage is opened at:
	https://github.com/simard-landscape-lab/kapok
     In this address above, they provide a document for install.  And for me, this command is worked:
	pip install git+https://github.com/mdenbina/kapok.git

(3)During installing of kapok, if have some error about GDAL package, this may be work:
	conda install gdal=3.0.2

(4) The input and output folder need to changed in the python file.

(5) The last verson is named "PIDLF+.py", which can directly run the code by :
	python PIDLF+.py   or 
	nohup python -u PIDLF+.py >v3.out&

