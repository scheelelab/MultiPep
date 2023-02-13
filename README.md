# MultiPep
### MultiPep stand-alone program and network parameters


<b>To use the MultiPep_predict.py for finding mean of models:</b>
`python MultiPep_predict.py -input_file test_seqs.txt -output_file output.csv -pred_type mean`

<br>

<b>To use the MultiPep_predict.py for finding max of models:</b>
`python MultiPep_predict.py -input_file test_seqs.txt -output_file output.csv -pred_type max`

<br>

<b>To geneate sequence data (not necessary):</b>
`python extract_data_and_bin_up2.py`

<br>

<p>final.py is the final version of MultiPep where the loss function is binary cross entropy and MCC.</p>
<p>weithed_final.py is the final version of MultiPep where the loss function is class weighted binary cross entropy.</p>

### News
The script `MultiPep_predict_xtra_output.py` has been added that allows users to use `-pred_type vote`, which outputs the number of models that have a prediction score > 0.5.
