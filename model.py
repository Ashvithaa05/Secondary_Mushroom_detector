import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping


# Load the dataset
file_path = r'primary_data.csv'  # Update this with the correct path
df = pd.read_csv(file_path, delimiter=';')

# Drop columns with many missing values or irrelevant to prediction
columns_to_drop = ['stem-root', 'veil-type', 'veil-color', 'Spore-print-color']
df = df.drop(columns=columns_to_drop)

# Handle missing values by imputing with the most frequent value (mode)
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encode categorical features using OneHotEncoding for better representation
categorical_columns = df_imputed.columns
encoder = OneHotEncoder(drop='first', sparse_output=False)
df_encoded = pd.DataFrame(encoder.fit_transform(df_imputed[categorical_columns]))
df_encoded.columns = encoder.get_feature_names_out(categorical_columns)

# Identify the new target columns corresponding to the original 'class' column
target_columns = [col for col in df_encoded.columns if 'class' in col]
X = df_encoded.drop(target_columns, axis=1)  # Features
y = df_encoded[target_columns]  # Target

# Feature Selection using SelectKBest to keep only the top features
selector = SelectKBest(chi2, k=15)  # Adjust 'k' to select the top features
X_selected = selector.fit_transform(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Normalize the features using MinMaxScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the deep neural network
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),  # Added Dropout
    Dense(128, activation='relu'),
    Dropout(0.5),  # Added Dropout
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming binary classification
])
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Adjusted learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history=model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
# Save the model
model.save('mushroom_model.h5')

# Save preprocessing objects
preprocessing_objects = {
    'imputer': imputer,
    'encoder': encoder,
    'selector': selector,
    'scaler': scaler
}

import pickle
with open('preprocessing_objects.pkl', 'wb') as f:
    pickle.dump(preprocessing_objects, f)