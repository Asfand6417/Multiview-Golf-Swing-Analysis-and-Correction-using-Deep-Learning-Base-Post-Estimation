# For classification tasks
y_pred = np.argmax(lstm_model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# For regression tasks
y_mpjpe_pred = mpjpe_model.predict(X_test).flatten()
mae = np.mean(np.abs(y_mpjpe_test - y_mpjpe_pred))
rmse = np.sqrt(np.mean((y_mpjpe_test - y_mpjpe_pred)**2))