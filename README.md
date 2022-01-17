# Journal of Computational Science (JCS) 2022
Codebase for "From Missing Data Imputation to Data Generation"

## Usage Example
<pre>
python main.py --algos="GAIN,SGAIN,WSGAIN-CP,WSGAIN-GP" --datasets="iris,yeast" --ampu_rate=0.2 
               --optimizer=GDA --learn_rate=0.001 
               --n_iterations=1000 --n_runs=3
</pre>

## Citing
<pre>
@article{dtneves:jcs:2022,
   title     = {{From Missing Data Imputation to Data Generation}},
   author    = {Diogo Telmo Neves, João Alves, Marcel Ganesh Naik, Alberto José Proença, and Fabian Prasser},
   booktitle = {Journal of Computational Science (JCS '22)},
   year      = {2022}
}
</pre>
