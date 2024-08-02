import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

file_path = 'E:\\staj\\customers-500000.csv'
df = pd.read_csv(file_path)
print(df.head())

df_grouped = df.groupby('Subscription Date').size().reset_index(name='Count')

df_grouped.to_csv('E:\\staj\\subscription_counts.csv', index=False)

unique_dates = df_grouped['Subscription Date'].nunique()
print(f"Total unique dates: {unique_dates}")

df_grouped.rename(columns={'Subscription Date': 'ds', 'Count': 'y'}, inplace=True)

df_grouped['ds'] = pd.to_datetime(df_grouped['ds'])
model = Prophet()
model.fit(df_grouped)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

forecast['ds'] = pd.to_datetime(forecast['ds'])

future_forecast = forecast[forecast['ds'] > df_grouped['ds'].max()]

combined = pd.concat([df_grouped[['ds', 'y']], future_forecast[['ds', 'yhat']].rename(columns={'yhat': 'y'})])
combined.to_csv('E:\\staj\\subscription_combined.csv', index=False)

future_forecast.to_csv('E:\\staj\\subscription_forecast.csv', index=False)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('E:\\staj\\subscription_forecast_full.csv', index=False)
plt.figure(figsize=(14, 7))
plt.plot(df_grouped['ds'], df_grouped['y'], 'o-', label='Actual Counts')
plt.xlabel('Дата')
plt.ylabel('Кол-во')
plt.title('До прогноза')
plt.legend()
plt.grid(True)
plt.savefig('E:\\staj\\subscription_counts_before_forecast.png')
plt.show()


plt.figure(figsize=(14, 7))
plt.plot(future_forecast['ds'], future_forecast['yhat'], 'o-', label='Predicted Counts')
plt.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'], alpha=0.2, label='Uncertainty Interval')
plt.xlabel('Дата')
plt.ylabel('Кол-во')
plt.title('После прогноза')
plt.legend()
plt.grid(True)
plt.savefig('E:\\staj\\subscription_counts_after_forecast.png')
plt.show()


plt.figure(figsize=(14, 7))
plt.plot(df_grouped['ds'], df_grouped['y'], 'o-', label='Actual Counts')
plt.plot(forecast['ds'], forecast['yhat'], 'x--', label='Predicted Counts', linestyle='dashed')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2, label='Uncertainty Interval')
plt.xlabel('Дата')
plt.ylabel('Колво')
plt.title('С начальными данными и прогнозом')
plt.legend()
plt.grid(True)
plt.savefig('E:\\staj\\subscription_counts_combined.png')
plt.show()


df.to_csv('E:\\staj\\original_customers.csv', index=False)


for date in df_grouped['ds'].unique():
    df_date = df_grouped[df_grouped['ds'] == date]
    count = df_date['y'].sum()
    print(f"Date: {date}, Count: {count}")

plt.figure(figsize=(14, 7))
plt.hist(df_grouped['y'], bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('Количество подписок')
plt.ylabel('Частота этого количества подписок')
plt.title('Гистограмма распределения количества подписок')
plt.grid(True)
plt.show()
