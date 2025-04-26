import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Custom Linear Regression class
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """Train the model using Gradient Descent"""
        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Calculate predictions
            y_pred = self.predict(X)
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate cost function and store for tracking
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
            
            # Print cost every 100 iterations
            if (i+1) % 100 == 0:
                print(f'Iteration {i+1}: Cost = {cost}')
        
        return self
    
    def predict(self, X):
        """Make predictions using the trained model"""
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """Calculate R² score of the model"""
        y_pred = self.predict(X)
        
        # Calculate Sum of Squared Errors (SSE)
        sse = np.sum((y - y_pred) ** 2)
        
        # Calculate Total Sum of Squares (SST)
        sst = np.sum((y - np.mean(y)) ** 2)
        
        # Calculate R² = 1 - SSE/SST
        r2 = 1 - (sse / sst)
        
        return r2

# Data preprocessing function
def preprocess_data(data):
    # Check data information
    print("Data information:")
    print(data.info())
    print("\nData statistics:")
    print(data.describe())
    
    # Check for null values
    null_values = data.isnull().sum()
    if null_values.sum() > 0:
        print("\nNumber of null values in each column:")
        print(null_values)
        print("\nHandling null values...")
        # You can add null value handling methods here
        data = data.dropna()  # Drop rows with null values
    
    # Split features and target
    X = data[['age', 'experience', 'education']].values
    y = data['salary'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Function to visualize results
def visualize_results(model, X, y, y_pred):
    plt.figure(figsize=(15, 10))
    
    # Cost Function plot
    plt.subplot(2, 2, 1)
    plt.plot(model.cost_history)
    plt.title('Cost Function over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    
    # Actual vs Predicted values
    plt.subplot(2, 2, 2)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title('Actual vs Predicted Salary')
    plt.xlabel('Actual Salary')
    plt.ylabel('Predicted Salary')
    
    # Error distribution
    plt.subplot(2, 2, 3)
    errors = y - y_pred
    plt.hist(errors, bins=30)
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    # Residual plot
    plt.subplot(2, 2, 4)
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.show()

# Education level mapping helper function
def get_education_level_description(level):
    education_levels = {
        0: "No Education",
        1: "Elementary School",
        2: "Middle School",
        3: "High School",
        4: "Bachelor's Degree",
        5: "Master's Degree",
        6: "PhD",
        7: "Professor"
    }
    return education_levels.get(level, "Unknown")

def main():
    # Read data from CSV file
    try:
        file_path = "salary_dataset.csv"  # Change this to your file path if needed
        print(f"Reading data from {file_path}...")
        data = pd.read_csv(file_path)
        print(f"Successfully read {len(data)} rows of data.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # Train the model
    print("\nTraining the Linear Regression model...")
    model = CustomLinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel evaluation:")
    print(f"R² on training set: {train_score:.4f}")
    print(f"R² on test set: {test_score:.4f}")
    
    # Calculate Mean Squared Error and Mean Absolute Error
    mse = np.mean((y_test - y_pred_test) ** 2)
    mae = np.mean(np.abs(y_test - y_pred_test))
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # Visualize results
    visualize_results(model, X_test, y_test, y_pred_test)
    
    # Display model weights
    print("\nModel weights:")
    feature_names = ['Bias', 'Age', 'Experience', 'Education']
    coefficients = [model.bias] + list(model.weights)
    
    for feature, coef in zip(feature_names, coefficients):
        print(f"{feature}: {coef:.4f}")
    
    # Function to predict salary based on input
    def predict_salary(age, experience, education):
        X_new = np.array([[age, experience, education]])
        X_new_scaled = scaler.transform(X_new)
        salary = model.predict(X_new_scaled)[0]
        return salary
    
    # Direct user input for salary prediction
    print("\nEnter information to predict salary:")
    while True:
        try:
            age = int(input("Enter age: "))
            experience = int(input("Enter years of experience: "))
            
            print("\nEducation levels:")
            for level in range(8):
                print(f"{level} - {get_education_level_description(level)}")
            
            education = int(input("Enter education level (0-7): "))
            if education < 0 or education > 7:
                print("Invalid education level. Please enter a number between 0 and 7.")
                continue
            
            salary_pred = predict_salary(age, experience, education)
            print(f"Predicted salary: ${salary_pred:.2f}")
            
            choice = input("\nDo you want to predict salary for another person? (y/n): ").lower()
            if choice != 'y':
                break
                
        except ValueError as e:
            print(f"Error: {e}. Please try again.")
        except KeyboardInterrupt:
            print("\nExiting program.")
            break

if __name__ == "__main__":
    main()