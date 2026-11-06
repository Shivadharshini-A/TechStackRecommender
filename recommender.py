
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Load dataset
def load_data():
    try:
        data = pd.read_csv("data/raw_skills.csv")
        return data
    except FileNotFoundError:
        print("Dataset not found!")
        exit()


# Step 2: Get user skills
def get_user_skills():
    print("\n===== TECH STACK RECOMMENDER =====")
    print("Enter at least 3 skills/interests.")

    skills = input("\nEnter your skills separated by commas: ")

    user_skills = [skill.strip().lower() for skill in skills.split(",")]
    user_skills = [skill for skill in user_skills if skill]

    if len(set(user_skills)) < 3:
        print("Please enter at least 3 different skills.")
        return get_user_skills()

    return user_skills


# Step 3: Generate recommendations
def recommend_roles(data, user_skills):

    # Convert job skills into text documents
    job_profiles = data["skills"].fillna("").tolist()

    # Create user profile
    user_profile = " ".join(user_skills)

    # Combine user and job profiles
    documents = job_profiles + [user_profile]

    # Apply TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Separate user vector
    user_vector = tfidf_matrix[-1]

    # Separate job vectors
    job_vectors = tfidf_matrix[:-1]

    # Calculate cosine similarity
    scores = cosine_similarity(user_vector, job_vectors)[0]

    # Store scores
    data = data.copy()
    data["Similarity Score"] = scores

    # Sort in descending order
    recommendations = data.sort_values(
        by="Similarity Score",
        ascending=False
    )

    # Return Top 3
    return recommendations.head(3)


# Step 4: Display results
def display_results(recommendations):

    print("\n========== TOP 3 CAREER RECOMMENDATIONS ==========")

    for index, (_, row) in enumerate(
        recommendations.iterrows(), start=1
    ):
        percentage = row["Similarity Score"] * 100

        print(f"\n{index}. {row['job_role']}")
        print(f"   Required Skills: {row['skills']}")
        print(f"   Match Score: {percentage:.2f}%")

    print("\n==================================================")


# Step 5: Main function
def main():

    data = load_data()

    user_skills = get_user_skills()

    print("\nYour Skills:", ", ".join(user_skills))

    recommendations = recommend_roles(data, user_skills)

    display_results(recommendations)


if __name__ == "__main__":
    main()
