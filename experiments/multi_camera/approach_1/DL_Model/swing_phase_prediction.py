# Replace the rule-based detect_swing_phase method
def detect_swing_phase(self):
    if len(self.lmList) < 33:
        return "Setup"  # Default when not enough data
        
    # Prepare features for the model
    features = prepare_features_for_model(self.lmList)
    
    # Make prediction using the trained model
    phase_probs = lstm_model.predict(np.array([features]))[0]
    phase_pred = np.argmax(phase_probs)
    
    # Map numerical prediction to phase name
    phase_names = ["Address", "Takeaway", "Backswing", "Top of Backswing", 
                  "Downswing", "Impact", "Follow Through"]
    return phase_names[phase_pred]