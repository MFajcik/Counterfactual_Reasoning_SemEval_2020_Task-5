import csv
roberta=set(['100001', '100042', '100137', '100300', '100380', '100448', '100453', '100508', '100549', '100574', '100580', '100944', '100969', '100977', '100985', '101020', '101085', '101175', '101238', '101296', '101526', '101536', '101607', '101612', '101635', '101810', '101871', '101890', '101899', '101929', '101930', '101942', '102102', '102116', '102140', '102181', '102230', '102304', '102367', '102391', '102441', '102470', '102473', '102484', '102528', '102673', '102757', '102828', '102888', '102901', '102960'])
albert=set(['100000', '100001', '100191', '100409', '100448', '100508', '100549', '100574', '100624', '100921', '100944', '100969', '100977', '100985', '100998', '101085', '101099', '101175', '101238', '101296', '101417', '101429', '101439', '101488', '101519', '101557', '101567', '101586', '101612', '101630', '101638', '101769', '101805', '101834', '101846', '101890', '101929', '101930', '101942', '101990', '102074', '102102', '102103', '102116', '102181', '102304', '102391', '102441', '102470', '102474', '102522', '102545', '102673', '102757', '102828', '102889', '102901', '102936', '102949'])
both_wrong=albert&roberta
albert_right_roberta_wrong=roberta-albert
roberta_right_albert_wrong=albert-roberta
with open("roberta_right_albert_wrong",'w') as rrawf,\
                        open("albert_right_roberta_wrong",'w')as arrwf, open("5a_both_wrong",'w') as bwf,\
                        open("/mnt/minerva1/nlp/projects/counterfactual/semeval/5/Subtask-1-master/dev.csv") as refs:
    lines_ref=[row for row in csv.reader(refs)]
    for ref in lines_ref:
        if ref[0] in both_wrong:
            bwf.write(','.join(ref)+'\n')
        if ref[0] in roberta_right_albert_wrong:
            rrawf.write(','.join(ref)+'\n')
        if ref[0] in albert_right_roberta_wrong:
            arrwf.write(','.join(ref)+'\n')