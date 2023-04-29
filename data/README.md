## Data Directory
Directory where data may be stored. An example of data folder structure might look like the following:
```
data/
|- input/
|	|- it/
|	|  |- train.tsv
|	|  |- test.tsv
|	|  |- dev.tsv
|	|- en_fce/
|	...
|
|- output/
|	|- it/
|	|  |- train.txt
|	|  |- test.txt
|	|  |- dev.txt
|	|  |- roberta_model/
|	|  |- roberta_preds_BIO.txt
|	|  |- it_preds.txt
|	|- en_fce/
|	...
|
|- output_all/
|	|- it/
|	|  |- train.txt
|	|  |- test.txt
|	|  |- dev.txt
|	|  |- roberta_model/
|	|  |- roberta_preds_BIO.txt
|	|  |- it_preds.txt
|	|- en_fce/
|	...
```