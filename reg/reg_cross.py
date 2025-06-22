import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import cross_val_predict

from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'data\\reg_data.csv' 
data = pd.read_csv(file_path, delimiter=';', skipinitialspace=True)


X = data.drop(columns=['year', 'врп'])
Y = data['врп']  
 

scaler = StandardScaler()
x_s = scaler.fit_transform(X)

# обучение 
lin_model = sm.OLS(Y, X).fit()

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(x_s, Y)

svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(x_s, Y)

poly = PolynomialFeatures(degree=3) 
X_poly = poly.fit_transform(x_s)
model_poly = LinearRegression().fit(X_poly, Y)

rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(x_s, Y)

s_lin_model = LinearRegression()
s_lin_model.fit(x_s, Y)


pred_val_poly = poly.transform(x_s)

predicted_Y_knn = knn_model.predict(x_s)
predicted_Y_svr = svr_model.predict(x_s)
predicted_Y_poly = model_poly.predict(pred_val_poly)
predicted_Y_rf = rf_model.predict(x_s)
predicted_Y_s_lin = s_lin_model.predict(x_s)


r2_knn = r2_score(Y, predicted_Y_knn)
r2_svr = r2_score(Y, predicted_Y_svr)
r2_poly = r2_score(Y, predicted_Y_poly)
r2_rf = r2_score(Y, predicted_Y_rf)
r2_s_lin = r2_score(Y, predicted_Y_s_lin)



cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_predictions = {}
metrics = {}

models = {
    'Linear Regression': s_lin_model,
    'Polynomial Regression (deg=3)': model_poly,
    'KNN': knn_model,
    'SVR': svr_model,
    'Random Forest': rf_model
}

X_poly_cv = poly.fit_transform(x_s)


feature_names = [f for f in X.columns if f != 'const']

print("\n📊 Кросс-валидация (R², MAE, RMSE):\n")

rf_importances = []

for name, model in models.items():
    if name == 'Polynomial Regression (deg=3)':
        X_use = X_poly_cv
    else:
        X_use = x_s

    if name == 'Random Forest':
        for train_idx, val_idx in cv.split(X_use):
            X_train, y_train = X_use[train_idx], Y.iloc[train_idx]
            rf_fold = RandomForestRegressor(n_estimators=100, random_state=0)
            rf_fold.fit(X_train, y_train)
            rf_importances.append(rf_fold.feature_importances_)

    preds = cross_val_predict(model, X_use, Y, cv=cv)
    cv_predictions[name] = preds

    mae = mean_absolute_error(Y, preds)
    rmse = np.sqrt(mean_squared_error(Y, preds))
    r2 = r2_score(Y, preds)

    metrics[name] = {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4)
    }
    metrics_df = pd.DataFrame(metrics).T  
  

mean_rf_importances = np.mean(rf_importances, axis=0)


poly_features = poly.get_feature_names_out(input_features=feature_names)
coefs_poly = model_poly.coef_

from collections import defaultdict
feature_importance_poly = defaultdict(float)

for feat_name, coef in zip(poly_features, coefs_poly):
    if feat_name == '1':  # игнорю константу
        continue

    parts = feat_name.split()
    for orig_feat in feature_names:
        if orig_feat in parts:
            feature_importance_poly[orig_feat] += abs(coef)


df_importance = pd.DataFrame({
    'Random Forest': mean_rf_importances,
    'Polynomial Regression': [feature_importance_poly.get(f, 0) for f in feature_names]
}, index=feature_names)


df_importance = df_importance.sort_values(by='Random Forest', ascending=False)




print(df_importance.round(4))



model_predictions = {
    'Linear Regression': predicted_Y_s_lin,
    'Polynomial Regression': predicted_Y_poly,
    'KNN': predicted_Y_knn,
    'SVR': predicted_Y_svr,
    'Random Forest': predicted_Y_rf
}

plt.figure(figsize=(12, 8))

years = data['year'].values


plt.plot(years, Y.values, 'k--', lw=2, label='Фактический (ВРП)')

for name, preds in cv_predictions.items():

    sorted_indices = np.argsort(years)
    years_sorted = years[sorted_indices]
    preds_sorted = preds[sorted_indices]

    plt.plot(years_sorted, preds_sorted, label=name, linewidth=1.5, marker='o', alpha=0.8)

plt.xlabel('Год', fontsize = 12)
plt.ylabel('ВРП (отклик)', fontsize = 12)
plt.title('Предсказания моделей по годам (кросс-валидация)', fontsize = 14)
plt.legend(fontsize = 12)
plt.grid(True)
plt.tight_layout()
plt.show()



models = metrics_df.index.tolist()
mae_values = metrics_df['MAE'].values
rmse_values = metrics_df['RMSE'].values
r2_values = metrics_df['R2'].values

x = np.arange(len(models))  
width = 0.35  

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# MAE и RMSE графики
bars1 = ax1.bar(x - width/2, mae_values, width, label='MAE (средняя абсолютная ошибка)', color='skyblue')
bars2 = ax1.bar(x + width/2, rmse_values, width, label='RMSE (среднеквадратическое отклонение)', color='lightgreen')

ax1.set_ylabel('Ошибка, тыс. руб.', fontsize = 12)
ax1.set_title('MAE и RMSE для каждой модели', fontsize = 14)
ax1.legend(fontsize = 11)
ax1.grid(True, linestyle='--', alpha=0.5)



# R**2 график
bars3 = ax2.bar(x, r2_values, width, label='R²', color='salmon')
max_r2 = max(r2_values)
ax2.set_ylim(-0.1, max_r2 + 0.1)  

ax2.set_ylabel('R²', fontsize = 12)
ax2.set_title('Коэффициент детерминации (R²)', fontsize = 14)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=30, fontsize = 12)
ax2.grid(True, linestyle='--', alpha=0.5)

for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}', 
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 4), textcoords='offset points',
                 ha='center', fontsize=10)
plt.tight_layout()
plt.show()




