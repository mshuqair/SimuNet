import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers.initializers_v2 import GlorotUniform
from keras_tuner import GridSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def build_model(hp):
    kernel_init = GlorotUniform(seed=1)
    optimizer = Adam(learning_rate=0.0001)

    layer_h1_units = hp.Choice(name='layer_h1_units', values=[128, 256, 512])
    # layer_h2_units = hp.Choice(name='layer_h2_units', values=[8, 16, 32, 64, 128])
    loss_function = hp.Choice(name='loss_function', values=['huber_loss'])

    layer_input = Input(shape=2, name='layer_input')
    layer_h1 = Dense(units=layer_h1_units, activation='relu', kernel_initializer=kernel_init, name='layer_h1')(
        layer_input)
    # layer_h2 = Dense(units=layer_h2_units, activation='relu', kernel_initializer=kernel_init, name='layer_h2')(layer_h1)
    layer_output = Dense(units=1, activation=None, kernel_initializer=kernel_init, name='layer_output')(layer_h1)

    # Construct the model
    model = Model(inputs=layer_input, outputs=layer_output)

    # Compile and model summary
    model.compile(optimizer=optimizer, loss=loss_function)
    model.summary()

    return model


# Main code
# Parameters
annual_maintenance_no = 720
years = np.array(object=[1, 2, 3, 4, 5], dtype=int)

# Planned cost (k_i)
a_vi = np.array([9.56, 10.516, 11.5676, 12.7244, 13.9968])
b_vi = np.array([3000, 3000, 3000, 3000, 3000])

# Unplanned cost (Linear)
a_s = np.array([6, 6.6, 7.26, 7.986, 8.7846])
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
    data[key]['Planned Cost'] = a_vi[year - 1] * data[key]['Maintenance No.'] + b_vi[year - 1]
    data[key]['Unplanned Cost'] = -a_s[year - 1] * data[key]['Maintenance No.'] + b_s[year - 1]
    data[key]['Total Cost'] = data[key]['Planned Cost'] + data[key]['Unplanned Cost']

# Splitting the data
# 0.8 for training and 0.2 for testing for each year
x_train, y_train = np.empty(shape=(0, 2)), np.empty(shape=(0, 1))
x_test, y_test = np.empty(shape=(0, 2)), np.empty(shape=(0, 1))

for year in years:
    year_no = data['Year ' + str(year)]['Year']
    maintenance_no = data['Year ' + str(year)]['Maintenance No.']
    total_cost = data['Year ' + str(year)]['Total Cost']
    x_temp = np.array([year_no, maintenance_no]).transpose()
    y_temp = np.array([total_cost]).transpose()
    x_train = np.append(x_train, x_temp[0:int(0.8 * annual_maintenance_no), :], axis=0)
    y_train = np.append(y_train, y_temp[0:int(0.8 * annual_maintenance_no), :], axis=0)
    x_test = np.append(x_test, x_temp[int(0.8 * annual_maintenance_no):, :], axis=0)
    y_test = np.append(y_test, y_temp[int(0.8 * annual_maintenance_no):, :], axis=0)

# Normalize data
scaler_std = StandardScaler()
x_train = scaler_std.fit_transform(x_train)
x_test = scaler_std.transform(x_test)

# Scale labels
scaler_minmax = MinMaxScaler(feature_range=(0, 1))
y_train = scaler_minmax.fit_transform(y_train)
y_test = scaler_minmax.transform(y_test)

# Split the data into training and validation percentage-wise
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

# using Grid Search method
tuner = GridSearch(hypermodel=build_model, objective='val_loss', max_trials=None,
                   directory='model optimization', project_name='simuNet', overwrite=True)

print('Summary of the search space:')
tuner.search_space_summary()
tuner.search(x=x_train, y=y_train, batch_size=32, epochs=15, validation_data=(x_valid, y_valid))

# summary of results
print('Summary of the search results:')
tuner.results_summary()

# best hyperparameters
print(print('Best hyperparameters:'))
best_hp = tuner.get_best_hyperparameters()[0]
print(best_hp.values)
