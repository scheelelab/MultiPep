# MultiPep
MultiPep stand-alone program and network parameters


To use the MultiPep_predict.py for finding mean of models:
<p>python MultiPep_predict.py -input_file test_seqs.txt -output_file output.csv -pred_type mean<p>


To use the MultiPep_predict.py for finding max of models:
<p>python MultiPep_predict.py -input_file test_seqs.txt -output_file output.csv -pred_type max<p>

To geneate sequence data:
<p>python extract_data_and_bin_up2.py<p>


<p>final.py is the final version of MultiPep where the loss function is binary cross entropy and MCC.<p>
<p>weithed_final.py is the final version of MultiPep where the loss function is class weighted binary cross entropy.<p>
