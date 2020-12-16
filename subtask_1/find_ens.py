import itertools,subprocess
import numpy as np
import taskA_scorer
from scipy.special import softmax

def all_subsets(ss):
    return itertools.chain(*map(lambda x: itertools.combinations(ss, x), range(0, len(ss)+1)))
    
#all_models=["models_bert/albert_4a_new_dev_nonaug/checkpoint-300/", "models_bert/albert_4a_new_dev_nonaug/checkpoint-400/", "models_bert/albert_4a_new_dev_nonaug/checkpoint-350", "models_bert/albert_4a_new_dev_aug2/checkpoint-400", "models_bert/albert_4a_new_dev_aug2/checkpoint-850/", "models_bert/albert_4a_new_dev_aug4/checkpoint-400/","models_bert/albert_4a_new_dev_aug4/checkpoint-1100/", "models_bert/albert_4a_new_dev_augmented_all_fixed/checkpoint-550/", "models_bert/albert_4a_new_dev_v1_2/checkpoint-800", "models_bert/albert_4a_new_dev_aug2/checkpoint-550/", "models_bert/albert_4a_new_dev_synth_cz2/checkpoint-500/", "models_bert/albert_4a_new_dev_synth_cz2/checkpoint-600/", "models_bert/albert_4a_new_dev_nonaug2/checkpoint-300/"]

#all_models=["eval_results_pred_nonaug300.npy","eval_results_pred_nonaug400.npy","eval_results_pred_nonaug2_400.npy","eval_results_pred_nonaug2_850.npy","eval_results_pred_aug4_400.npy", "eval_results_pred_aug_all550.npy"]
all_models= ["eval_results_pred_trial_nonaug2_850.npy", "eval_results_pred_trial_aug4_400.npy","eval_results_pred_trial_nonaug2_400.npy","eval_results_pred_trial_nonaug300.npy","eval_results_pred_trial_nonaug400.npy","eval_results_pred_trial_aug4_1100.npy", "eval_results_pred_trial_nonaug4_350.npy", "eval_results_pred_trail_aug2_400.npy", "eval_results_pred_trial_cz2_500.npy"]

model_combinations=all_subsets(all_models)
best_acc=0
best_comb=[]
for comb in model_combinations:
	probs=np.zeros((997,2),dtype=np.float64)
	probs=np.zeros((2021,2),dtype=np.float64)

	results=[]
	for model in comb:
#		results.append(model+"/eval_results_pred.npy")
		results.append("model_eval/"+model)
	print(results)
	#print(probs)
	for f in results:
		#print("adding %s"%f)
		probs+=softmax(np.load(f),axis=1)
	#print(probs)
	with open("tmp_res.txt","w") as res:
		for i,p in enumerate(probs):
			res.write("{},{}\n".format(i+1,np.argmax(p)))
#	gold_labels = taskA_scorer.read_gold("subtaskA_gold_answers_reid.csv")
	gold_labels = taskA_scorer.read_gold("taskA_trial_answer.csv")

	pred_labels =  taskA_scorer.read_predictions("tmp_res.txt")
	accuracy =  taskA_scorer.calculate_accuracy(gold_labels, pred_labels)
	if accuracy>best_acc:
		best_acc=accuracy
		best_comb=results
	print(accuracy)
print("best ensemble:")
print(best_comb)
print(best_acc)
	#print(subprocess.run(["python3" "taskA_scorer.py", "-p", "tmp_res.txt", "-g" "subtaskA_gold_answers_reid.csv"], capture_output=True,shell=True))
