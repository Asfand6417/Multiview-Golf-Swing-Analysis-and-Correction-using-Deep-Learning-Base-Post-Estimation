# For the LSTM swing phase classifier
history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('models/lstm_golf_swing.h5', monitor='val_loss', save_best_only=True)
    ]
)

# For the pose error regression model
history_mpjpe = mpjpe_model.fit(
    X_train, y_mpjpe_train,
    validation_data=(X_val, y_mpjpe_val),
    epochs=30,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('models/mpjpe_golf_swing.h5', monitor='val_loss', save_best_only=True)
    ]
)