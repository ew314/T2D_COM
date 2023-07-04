T2D_COM
================================
# System Requirements and Installation
---------------------------------
Python == 3.9.1<br>
numpy == 1.16.5<br>
tensorflow == 2.12.0<br>
keras == 2.12.0<br>
scipy == 1.2.1<br>
sklearn == 0.20.3<br>
matplotlib == 2.2.3<br>

---
Getting Started
---------------
Put script 'net_work_final.py' and folder 'train_data' in same path

Usage Examples
--------------

    python3 net_work_final.py

Output
--------------
      ./final_model_logit.ROC_AUC.png/PR.png
      ./final_model_RF.ROC_AUC.png/PR.png
      ./final_model_SVM.ROC_AUC.png/PR.png
      ./final_model_model.ROC_AUC.png/PR.png
      ./T2D_final_model.h5

`-ROC_AUC.png/PR.png`                                         : ROC-AUC (Receiver Operating Characteristic, Area Under Curve) and PR-AUC (Precision Recall Curve, Area Under Curve) of the logistic regression(Logit), random forest(RF), support vector machine(SVM) and artificial neural network(final_model_model) model<br>
`T2D_final_model.h5`: Trained model<br>

---
