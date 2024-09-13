from keras.layers import Dense, Input
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.initializers.initializers_v2 import GlorotUniform
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# The data generated using a reciprocal function
# The network is trained on 0.8 of the data and tested on 0.2


def build_model():
    optimizer = Adam(learning_rate=learning_rate)
    kernel_init = GlorotUniform(seed=1)
    model = Sequential()
    model.add(Input(shape=2))
    model.add(Dense(units=512, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(units=1, activation=None, kernel_initializer=kernel_init))
    model.compile(optimizer=optimizer, loss='huber_loss')
    model.summary(show_trainable=True)
    return model


def plot_data(data):
    for key in data.keys():
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 3), layout='constrained')
        ax.set_title('Maintenance cost - ' + key)
        ax.plot(data[key]['Maintenance No.'], data[key]['Planned Cost'], label='Planned Cost')
        ax.plot(data[key]['Maintenance No.'], data[key]['Unplanned Cost'], label='Unplanned Cost')
        ax.plot(data[key]['Maintenance No.'], data[key]['Total Cost'], label='Total Cost')
        ax.set_ylim(0, 25000)
        ax.set_ylabel('Euros')
        ax.set_xlabel('Number of maintenance  procedures')
        ax.legend(loc='best', fontsize='small')
        plt.show()


# Main code
sns.set_theme(style='darkgrid')

epochs = 50
batch_size = 32
learning_rate = 0.0001

# Parameters
annual_maintenance_no = 720
years = np.array(object=[1, 2, 3, 4, 5], dtype=int)

# Planned cost (k_i)
a_vi = np.array([9.56, 10.516, 11.5676, 12.7244, 13.9968])
b_vi = np.array([3000, 3000, 3000, 3000, 3000])

# Unplanned cost (Reciprocal)
a_s = np.array([5600, 5384.615, 5177.515, 4978.38, 4786.903])
b_s = np.array([7200, 7200, 7200, 7200, 7200])


# Create the data dictionary
data = {}
for year in years:
    data['Year ' + str(year)] = {'Year': np.array([]),
                                 'Maintenance No.': np.array([]),
                                 'Planned Cost': np.array([]),
                                 'Unplanned Cost': np.array([]),
                                 'Total Cost': np.array([])}

# Generate data
for key, year in zip(data.keys(), years):
    data[key]['Year'] = np.full(shape=annual_maintenance_no, fill_value=year)
    data[key]['Maintenance No.'] = np.arange(annual_maintenance_no) + 1
    data[key]['Planned Cost'] = a_vi[year-1]*data[key]['Maintenance No.'] + b_vi[year-1]
    data[key]['Unplanned Cost'] = a_s[year-1]/data[key]['Maintenance No.'] + b_s[year-1]
    data[key]['Total Cost'] = data[key]['Planned Cost'] + data[key]['Unplanned Cost']

plot_data(data)

# Splitting the data
# 0.8 for training and 0.2 for testing for each year
x_train, y_train = np.empty(shape=(0, 2)), np.empty(shape=(0, 1))
x_test, y_test = np.empty(shape=(0, 2)), np.empty(shape=(0, 1))

for year in years:
    year_no =  data['Year ' + str(year)]['Year']
    maintenance_no = data['Year ' + str(year)]['Maintenance No.']
    total_cost = data['Year ' + str(year)]['Total Cost']
    x_temp = np.array([year_no, maintenance_no]).transpose()
    y_temp = np.array([total_cost]).transpose()
    x_train = np.append(x_train, x_temp[0:int(0.8*annual_maintenance_no), :], axis=0)
    y_train = np.append(y_train, y_temp[0:int(0.8*annual_maintenance_no), :], axis=0)
    x_test = np.append(x_test, x_temp[int(0.8*annual_maintenance_no):, :], axis=0)
    y_test = np.append(y_test, y_temp[int(0.8*annual_maintenance_no):, :], axis=0)


# Normalize data
scaler_std = StandardScaler()
x_train = scaler_std.fit_transform(x_train)
x_test = scaler_std.transform(x_test)

# Scale labels
scaler_minmax = MinMaxScaler(feature_range=(0, 1))
y_train = scaler_minmax.fit_transform(y_train)
y_test = scaler_minmax.transform(y_test)

# Train model if not trained, otherwise load it
if not os.path.isfile('models/model_nn_reciprocal_split.h5'):
    model_nn = build_model()
    callback_stopping = EarlyStopping(monitor='loss', patience=5)
    history = model_nn.fit(x_train, y_train,
                           batch_size=batch_size, epochs=epochs,
                           callbacks=[callback_stopping])
    model_nn.save('models/model_nn_reciprocal_split.h5')
else:
    model_nn = load_model('models/model_nn_reciprocal_split.h5')


# Evaluate
y_predicted = model_nn.predict(x_test)
y_predicted = scaler_minmax.inverse_transform(y_predicted)
y_test = scaler_minmax.inverse_transform(y_test)
y_train = scaler_minmax.inverse_transform(y_train)

# Metrics for all years
RMSE = root_mean_squared_error(y_true=y_test, y_pred=y_predicted)
MAE = mean_absolute_error(y_true=y_test, y_pred=y_predicted)
r_2 = r2_score(y_true=y_test, y_pred=y_predicted)
corr = pearsonr(x=y_test, y=y_predicted)
print('Overall Metrics:')
print('RMSE %.2f, MAE %.2f, R^2 score %.2f, Correlation coefficient %.2f (p=%.4f)'
      % (RMSE, MAE, r_2, corr[0][0], corr[1][0]))


# Metrics for each year
# Split into 5 arrays 1 for each year
y_predicted = np.split(y_predicted, indices_or_sections=5)
y_test = np.split(y_test, indices_or_sections=5)
y_train = np.split(y_train, indices_or_sections=5)

for year, year_index in zip(years, years-1):
    RMSE = root_mean_squared_error(y_true=y_test[year_index], y_pred=y_predicted[year_index])
    MAE = mean_absolute_error(y_true=y_test[year_index], y_pred=y_predicted[year_index])
    r_2 = r2_score(y_true=y_test[year_index], y_pred=y_predicted[year_index])
    corr = pearsonr(x=y_test[year_index], y=y_predicted[year_index])
    print('Metrics for Year: ' + str(year))
    print('RMSE %.2f, MAE %.2f, R^2 score %.2f, Correlation coefficient %.2f (p=%.4f)'
          % (RMSE, MAE, r_2, corr[0][0], corr[1][0]))

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 3), layout='constrained')
    ax.set_title('Total Cost Year ' + str(year))
    ax.plot(data['Year ' + str(year)]['Maintenance No.'],
            np.concatenate((y_train[year_index], y_test[year_index]), axis=0),
            label='Actual')
    ax.plot(data['Year ' + str(year)]['Maintenance No.'][int(0.8*annual_maintenance_no):],
            y_predicted[year_index],
            linestyle='--', label='Predicted')
    ax.axvline(x=int(0.8*annual_maintenance_no),  ymin=0, ymax=1, linestyle='dotted', linewidth=1, color='grey')
    ax.set_ylim(0, 25000)
    ax.set_ylabel('Euros')
    ax.set_xlabel('Number of maintenance  procedures')
    ax.legend(loc='best', fontsize='small')
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 3), layout='constrained')
    ax.set_title('Network Error - Year ' + str(year))
    ax.plot(data['Year ' + str(year)]['Maintenance No.'][int(0.8*annual_maintenance_no):],
            y_test[year_index]-y_predicted[year_index])
    ax.set_ylabel('Euros')
    ax.set_xlabel('Number of maintenance  procedures')
    plt.show()
