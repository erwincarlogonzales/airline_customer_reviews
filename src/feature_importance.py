import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Train model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    
    return model

# Get feature importance
def get_feature_importance(model, feature_names):
    
    importance = model.feature_importances_
        
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False)
    
    return feature_importance

# Plot feature importance
def plot_feature_importance(feature_importance, title):
    
    plt.figure(figsize=(10, 6))
    
    # Barplot
    sns.barplot(
        data=feature_importance,
        x='importance',
        y='feature',
        palette='Spectral_r',
        hue='feature'
    )
    
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()