def create_pose_error_model(input_shape):
    input_layer = Input(shape=input_shape)
    
    x = LSTM(128, return_sequences=True)(input_layer)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='linear')(x)  # Regression output (MPJPE)
    
    model = Model(inputs=input_layer, outputs=x)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model