from data.generate_emails import generate_emails
from models.train_naive_bayes import train_naive_bayes
from models.train_logistic_regression import train_logistic_regression
from models.train_random_forest import train_random_forest
from models.train_svm import train_svm
from models.train_bert import train_bert
from evaluation.evaluate import evaluate_models
from utils.utils import print_separator

def main():
    print_separator()
    print("Step 1: Generating Email Dataset")
    generate_emails()

    print_separator()
    print("Step 2: Training Models")
    train_naive_bayes()
    train_logistic_regression()
    train_random_forest()
    train_svm()
    train_bert()

    print_separator()
    print("Step 3: Evaluating Models")
    evaluate_models()

if __name__ == "__main__":
    main()
