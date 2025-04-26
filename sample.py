import numpy as np
import pandas as pd

def generate_salary_data(n_samples=1000, filename="salary_dataset.csv", min_salary=100, max_salary=3000):
    """Generate sample data and save to CSV file with salary range 100-3000 USD"""
    np.random.seed(42)
    
    # Generate random data with reasonable ranges
    age = np.random.randint(18, 65, size=n_samples)  # Age range from 18-65
    
    # Work experience: related to age but with some variation
    experience = np.maximum(0, age - np.random.randint(18, 25, size=n_samples))
    
    # Education level:
    # 0 - No education
    # 1 - Elementary school
    # 2 - Middle school
    # 3 - High school
    # 4 - Bachelor's degree
    # 5 - Master's degree
    # 6 - PhD
    # 7 - Professor
    education = np.random.randint(0, 8, size=n_samples)
    
    # Adjust coefficients to keep salary within 100-3000 USD range
    base_salary = 100       # Minimum base salary
    exp_coef = 30           # Experience coefficient
    edu_coef = 250          # Education level coefficient
    age_coef = 2            # Age coefficient
    
    # Calculate salary using the adjusted formula
    noise = np.random.normal(0, 100, size=n_samples)
    raw_salary = base_salary + exp_coef * experience + edu_coef * education + age_coef * age + noise
    
    # Ensure salary is within the min_salary to max_salary range
    salary = np.clip(raw_salary, min_salary, max_salary)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'experience': experience,
        'education': education,
        'salary': salary
    })
    
    # Save to CSV file
    data.to_csv(filename, index=False)
    print(f"Generated and saved {n_samples} records to {filename}")

if __name__ == "__main__":
    # Generate 1000 data samples and save to salary_dataset.csv
    generate_salary_data(n_samples=1000)