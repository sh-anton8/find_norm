from tools import xgboost_model, xgboost_predict, xgboost_features

xgboost_features.features_to_files(3)
xgb_model = xgboost_model.train_xgboost_model()
xgboost_predict.predict_xgboost_answers(xgb_model)