import csv,sys
predsf='subtask1_pred_heldout.csv'
#predsf='albert_best_preds_5a_heldout.csv'
#predsf='subtask1_albert+roberta_pred_heldout.csv'
reff='/mnt/minerva1/nlp/projects/counterfactual/semeval/5/Subtask-1-master/dev.csv'
wrong_ids=[]
false_neg_cnt=0
false_pos_cnt=0
with open(predsf) as preds, open(reff) as refs:
    lines_preds=[row for row in csv.reader(preds)]
    lines_ref=[row for row in csv.reader(refs)]
    for pred,ref in zip(lines_preds,lines_ref):
        if pred[0]!=ref[1]:
            wrong_ids.append(ref[0])
            if ref[1]=='0':
                false_pos_cnt+=1
            else:
                false_neg_cnt+=1
            print(ref)
print(wrong_ids)
sys.stderr.write("errors: %s" %str(false_neg_cnt+false_pos_cnt))
sys.stderr.write("false neg: %s" %str(false_neg_cnt))
sys.stderr.write("false pos: %s" %str(false_pos_cnt))

