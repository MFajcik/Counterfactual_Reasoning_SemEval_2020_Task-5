## BUT-FIT at SemEval-2020 Task 5:  Automatic detection of counterfactual statements with deep pre-trained language representation models
__Authors__:
* Martin Fajčík
* Josef Jon
* Martin Dočekal
* Pavel Smrž

This is a official implementation we have used in the SemEval-2020 Task 5. Our publication is available [here](https://www.aclweb.org/anthology/2020.semeval-1.53/).
All models have been trained on RTX 2080 Ti (with 12 GB memory).

## Paper abstract
This paper describes BUT-FIT’s submission at SemEval-2020 Task 5: Modelling Causal Reasoning in Language: Detecting Counterfactuals. The challenge focused on detecting whether a given statement contains a counterfactual (Subtask 1) and extracting both antecedent and consequent parts of the counterfactual from the text (Subtask 2). We experimented with various state-of-the-art language representation models (LRMs). We found RoBERTa LRM to perform the best in both subtasks. __We achieved the first place in both exact match and F1 for Subtask 2 and ranked second for Subtask 1.__

## Bibtex citation
```
@inproceedings{fajcik2020but,
  title={BUT-FIT at SemEval-2020 Task 5: Automatic Detection of Counterfactual Statements with Deep Pre-trained Language Representation Models},
  author={Fajcik, Martin and Jon, Josef and Docekal, Martin and Smrz, Pavel},
  booktitle={Proceedings of the Fourteenth Workshop on Semantic Evaluation},
  pages={437--444},
  year={2020}
}
```

Please see README's in individual subfolders `subtask_1` and `subtask_2` for further details.
