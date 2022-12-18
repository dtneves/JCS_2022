# Journal of Computational Science (JCS) 2022
Codebase for "From Missing Data Imputation to Data Generation"

## Usage Example
<pre>
python main.py --algos="CTGAN,tabulator,tabulator-CP,tabulator-GP" --datasets="iris,yeast" --ampu_rates="0.2,0.4,0.6" 
               --optimizer=GDA --learn_rate=0.001 
               --n_iterations=1000 --n_runs=3
</pre>

## Citing
<pre>
@article{neves2022missing,
  title={From Missing Data Imputation to Data Generation},
  author={Neves, Diogo Telmo and Alves, Jo{\~a}o and Naik, Marcel Ganesh and Proen{\c{c}}a, Alberto Jos{\'e} and Prasser, Fabian},
  journal={Journal of Computational Science},
  volume={61},
  year={2022},
  publisher={Elsevier}
}
</pre>
