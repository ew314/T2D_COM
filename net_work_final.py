from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from sklearn import  linear_model,metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc as sklearn_auc
#######  keras #########
import keras,os
from keras.layers import Input, Conv1D, Conv2D, Activation, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, ZeroPadding2D
from keras.models import load_model
from keras.models import Model
from keras.constraints import max_norm
####### matplotlib ######## 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig

globeloc=os.getcwd()

def Confusion_Matrix(true_label,pre_label):
	TP=float(list(true_label+pre_label).count(2))
	FN=float(list(true_label-pre_label).count(1))
	FP=float(list(true_label-pre_label).count(-1))
	TN=float(list(true_label+pre_label).count(0))
	try:
		acc=(TP+TN)/(TP+FN+FP+TN)
	except:
		acc=0
	try:
		pre=(TP)/(TP+FP)
	except:
		pre=0
	try:
		tpr=(TP)/(TP+FN)
	except:
		tpr=0
	try:
		tnr=(TN)/(TN+FP)
	except:
		tnr=0
	try:
		F1=(2*pre*tpr)/(pre+tpr)
	except:
		F1=0
	return TP,FN,FP,TN,acc,pre,tpr,tnr,F1


def model_check(true_label,pre_label,pre_value):
	TP,FN,FP,TN,acc,pre,tpr,tnr,F1=Confusion_Matrix(true_label,pre_label)
	fpr_list, tpr_list, _ = metrics.roc_curve(true_label,pre_value,pos_label=1)	
	roc_auc=metrics.auc(fpr_list, tpr_list)	
	lr_precision, lr_recall, _ = precision_recall_curve(true_label,pre_value,pos_label=1)	
	PR_auc=sklearn_auc(lr_recall, lr_precision)
	return TP,FN,FP,TN,acc,pre,tpr,tnr,F1,roc_auc,PR_auc	

def mini_bathc(X_train,y_train):
	
	indices=np.arange(len(y_train))
	np.random.shuffle(indices)	
	tp_y_train=y_train[indices]
	tp_X_train=X_train[indices]
	y_label=tp_y_train[:,1]
	max_num=min([list(y_label).count(0),list(y_label).count(1)])
	pos_ind=np.argwhere(y_label==1)[:max_num]
	neg_ind=np.argwhere(y_label==0)[:max_num]
	tp_new_y=np.append(tp_y_train[pos_ind][:,0],tp_y_train[neg_ind][:,0],axis=0)
	tp_new_x=np.append(tp_X_train[pos_ind][:,0],tp_X_train[neg_ind][:,0],axis=0)
	indices=np.arange(len(tp_new_y))
	np.random.shuffle(indices)		
	mini_y_train=tp_new_y[indices]
	mini_x_train=tp_new_x[indices]
	
	return mini_x_train,mini_y_train


def data_load(file_name,X_train,y_train,factor_group):
	f1=open(file_name,'r')
	m1=f1.readlines()
	f1.close()
	for i in range(1,len(m1)):
		p1=m1[i].strip().split('\t')
		tp=[]
		for j in factor_group:
			tp.append(float(p1[j]))
		X_train.append(np.array(tp))
		if p1[-1]=='1':
			y_train.append([0,1])
		else:
			y_train.append([1,0])
	return X_train,y_train

def socre_label(pre_score):
	pre_label=[]
	for tp in pre_score:
		if tp[0] > tp[1]:
			pre_label.append(0)
		else:
			pre_label.append(1)
	return pre_label

def data_load(file_name,X_train,y_train,factor_group):
	f1=open(file_name,'r')
	m1=f1.readlines()
	f1.close()
	for i in range(1,len(m1)):
		p1=m1[i].strip().split('\t')
		tp=[]
		for j in factor_group:
			tp.append(float(p1[j]))
		X_train.append(np.array(tp))
		y_train.append(int(p1[-1]))
	return X_train,y_train


def data_load_model(file_name,X_train,y_train,factor_group):
	f1=open(file_name,'r')
	m1=f1.readlines()
	f1.close()
	for i in range(1,len(m1)):
		p1=m1[i].strip().split('\t')
		tp=[]
		for j in factor_group:
			tp.append(float(p1[j]))
		X_train.append(np.array(tp))
		if p1[-1]=='1':
			y_train.append([0,1])
		else:
			y_train.append([1,0])
	return X_train,y_train

def plot_AUC(y_true,y_score,figname):
	########## ROC AUC  #################
	fpr, tpr, thresholds_keras = metrics.roc_curve(y_true,y_score,pos_label=1)	
	auc=metrics.auc(fpr, tpr)	
	print("AUC : ", auc)
	plt.figure(figsize=(6,5))
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.savefig(figname+'ROC_AUC.png', dpi=300)
	plt.close('all')
	plt.clf()	
	lr_precision, lr_recall, _ = precision_recall_curve(y_true,y_score,pos_label=1)	
	lr_auc=sklearn_auc(lr_recall, lr_precision)
	print('PR-AUC=%.3f' %(lr_auc))
	plt.figure(figsize=(5,5))
	plt.plot(lr_recall, lr_precision,label='S3< val (AUC = {:.3f})'.format(lr_auc))
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='best')
	plt.savefig(figname+'PR.png', dpi=300)
	plt.close('all')
	plt.clf()			



file1='%s/train_data/5_fold_1.txt'%globeloc
file2='%s/train_data/5_fold_2.txt'%globeloc
file3='%s/train_data/5_fold_3.txt'%globeloc
file4='%s/train_data/5_fold_4.txt'%globeloc
file5='%s/train_data/5_fold_5.txt'%globeloc
file6='%s/train_data/vaildation_data.txt'%globeloc


all_factor=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,26,27,28]

data_ind=all_factor


X_train=[]
y_train=[]
	
X_train,y_train=data_load(file1,X_train,y_train,data_ind)
X_train,y_train=data_load(file2,X_train,y_train,data_ind)
X_train,y_train=data_load(file3,X_train,y_train,data_ind)
X_train,y_train=data_load(file4,X_train,y_train,data_ind)
X_train,y_train=data_load(file5,X_train,y_train,data_ind)

X_val=[]
y_val=[]
X_val,y_val=data_load(file6,X_val,y_val,data_ind)

X_train=np.array(X_train)
y_train=np.array(y_train)
X_val=np.array(X_val)
y_val=np.array(y_val)





clf1 = LogisticRegression(random_state=0).fit(X_train, y_train)
clf2 = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
clf3 = svm.SVC(probability=True).fit(X_train, y_train)

y_predict1=clf1.predict(X_val)
y_predict_value1=clf1.predict_proba(X_val)[:,1]
		
y_predict2=clf2.predict(X_val)
y_predict_value2=clf2.predict_proba(X_val)[:,1]
		
y_predict3=clf3.predict(X_val)
y_predict_value3=clf3.predict_proba(X_val)[:,1]	
		
TP,FN,FP,TN,acc,pre,tpr,tnr,F1,roc_auc,PR_auc	=model_check(y_val,y_predict1,y_predict_value1)

figname='%s/final_model_logit'%globeloc
plot_AUC(y_val,y_predict_value1,figname)


TP,FN,FP,TN,acc,pre,tpr,tnr,F1,roc_auc,PR_auc	=model_check(y_val,y_predict2,y_predict_value2)

figname='%s/final_model_RF'%globeloc
plot_AUC(y_val,y_predict_value2,figname)



TP,FN,FP,TN,acc,pre,tpr,tnr,F1,roc_auc,PR_auc	=model_check(y_val,y_predict3,y_predict_value3)

figname='%s/final_model_SVM'%globeloc
plot_AUC(y_val,y_predict_value3,figname)



X_train=[]
y_train=[]
	
X_train,y_train=data_load_model(file1,X_train,y_train,data_ind)
X_train,y_train=data_load_model(file2,X_train,y_train,data_ind)
X_train,y_train=data_load_model(file3,X_train,y_train,data_ind)
X_train,y_train=data_load_model(file4,X_train,y_train,data_ind)
X_train,y_train=data_load_model(file5,X_train,y_train,data_ind)

X_val=[]
y_val=[]
X_val,y_val=data_load_model(file6,X_val,y_val,data_ind)

X_train=np.array(X_train)
y_train=np.array(y_train)
X_val=np.array(X_val)
y_val=np.array(y_val)



sample_weights_c=[]
class_weights_c={}
class_number_c=[0,0]
for i in range(0,len(y_train)):
	if y_train[i][1] == 1:
		class_number_c[1]=class_number_c[1]+1
	if y_train[i][1] == 0:
		class_number_c[0]=class_number_c[0]+1

class_bin_c=[1/float(class_number_c[0]),1/float(class_number_c[1])]
class_weights_c['0']=class_bin_c[0]/sum(class_bin_c)
class_weights_c['1']=class_bin_c[1]/sum(class_bin_c)

for i in range(0,len(y_train)):
	if y_train[i][1] ==1:
		sample_weights_c.append(class_weights_c['1'])
	else:
		sample_weights_c.append(class_weights_c['0'])

sample_weights_c=np.array(sample_weights_c)


batch_size=8
epoch=256


## model ##
input_all = Input(shape=(len(X_train[0])))
Dense1 = Dense(128,kernel_regularizer=keras.regularizers.l1_l2(0.0001, 0.0001),activation='hard_sigmoid')(input_all)
Dropout1 = Dropout(0.25)(Dense1)
Batch1 = BatchNormalization()(Dropout1)
Dense2 = Dense(64,kernel_regularizer=keras.regularizers.l1_l2(0.0001, 0.0001),activation='hard_sigmoid')(Batch1)
Dropout2 = Dropout(0.25)(Dense2)
Batch2 = BatchNormalization()(Dropout2)
output = Dense(2, activation='softmax')(Batch2)
model_class = Model(inputs=input_all, outputs=output)

adam=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
model_class.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=adam)
model_class.fit(X_train, y_train, sample_weight=np.array(sample_weights_c),batch_size=batch_size, epochs=epoch, validation_data=(X_val, y_val),verbose=0)

mp = "%s/T2D_final_model.h5"%globeloc
model_class.save(mp)

pre_score=model_class.predict(X_val)
pre_label=socre_label(pre_score)
y_val_label=np.array(socre_label(y_val))
TP,FN,FP,TN,acc,pre,tpr,tnr,F1,roc_auc,PR_auc=model_check(y_val_label,pre_label,pre_score[:,1])

figname='%s/final_model_model'%globeloc
plot_AUC(y_val_label,pre_score[:,1],figname)









	
	