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
- The script `MultiPep_predict_xtra_output.py` has been added that allows users to use `-pred_type vote`, which outputs the number of models that have a prediction score > 0.5.

- The script `MultiPep_precision.py` checks the number of false positives and the precision scores for all classes under the thresholds: 0.5, 0.7 and 0.9. We observe a significant increase in precision scores and a significant decrease in number of false positives when comparing the results for thresholds 0.5 vs. 0.7 and 0.5 vs. 0.9. This is true for both the validation sets and test sets for alle CV models. The Wilcoxon signed-rank test was used to calculate the p-values, and the p-values where corrected for multiple testing using Benjamini/Hochberg method with an alpha = 0.01. 

- Note that the opposite tendensies are true for the recall score and number of false negatives. This can be calculated in a similar manner using the script `MultiPep_recall.py`.
