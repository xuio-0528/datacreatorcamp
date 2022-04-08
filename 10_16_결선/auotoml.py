## Regression 일 경우
reg = setup(session_id=1,
            data=X_train_ln,
            target='중식계',
            #numeric_imputation = 'mean',
            normalize = True,
            #categorical_features=['월', '요일', '공휴일전후'],
            silent=True)
best_5_l = compare_models(sort='MAE', n_select=5)
blended_l = blend_models(estimator_list= best_5_l, fold=5, optimize='MAE')
pred_holdout = predict_model(blended_l)
final_model_l = finalize_model(blended_l)
pred_esb_l = predict_model(final_model_l, test_data)

## Classification 일 경우(:class 여러가지, probability계산)
from pycaret.classification import *
from sklearn.metrics import log_loss

#아래 코드에서는 logloss로 loss를 계산합니다
clf = setup(train, target = target, train_size = 0.85)
add_metric('logloss', 'LogLoss', log_loss, greater_is_better=False, target="pred_proba")
best5 = compare_models(fold = 5, sort = 'logloss', n_select = 5, exclude=['svm','ridge'])
blended = blend_models(estimator_list = best5, fold = 5, optimize = 'logloss')
pred_holdout = predict_model(blended)
final_model = finalize_model(blended)
#Accurary, AUC, Logloss 셋다 상위 4개인 모델 사용
prep_pipe = get_config("prep_pipe")
prep_pipe.steps.append(['trained_model', final_model])
prections = prep_pipe.predict_proba(test)
prections