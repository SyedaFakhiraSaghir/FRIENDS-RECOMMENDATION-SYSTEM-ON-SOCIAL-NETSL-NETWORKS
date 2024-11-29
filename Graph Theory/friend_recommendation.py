import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the dataset
file_path = "/content/friend_network.csv"  # Replace with the correct file path
df = pd.read_csv(file_path)

# Step 1.1: Check the column names in the CSV file and adjust if needed
print("Column Names in Dataset:")
print(df.columns)

# Since your data has no 'Node1' and 'Node2', let's create some relationships:
# Example: Let's assume that people with the same location or hobbies are "friends"
# We will create edges between people who share a common location or hobby.

# Step 2: Create a graph based on shared location or hobbies
G = nx.Graph()

# Add edges based on Friends column
for index, row in df.iterrows():
    person_name = row['Name']
    friends_list = row['Friends']

    # Check if friends_list is a string and convert it to a list of integers if needed
    if isinstance(friends_list, str):
        # Remove brackets and split the string into individual friend IDs
        friends_list = [int(x) for x in friends_list.strip('[]').split(',')]

    # Assuming 'Friends' column contains a list of friend IDs
    for friend_id in friends_list:
        # Find the friend's name based on their ID
        # Check if the friend_id exists in the 'ID' column
        if friend_id in df['ID'].values:
            friend_name = df.loc[df['ID'] == friend_id, 'Name'].iloc[0]
            # Add an edge between the person and their friend
            G.add_edge(person_name, friend_name)
        else:
            print(f"Warning: Friend ID {friend_id} not found in the dataset for {person_name}. Skipping this edge.")


# Step 3: Visualize the graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=2.0, iterations=50)
nx.draw(
    G, pos,
    with_labels=True,
    node_size=300,
    node_color="lightblue",
    font_size=10,
    font_weight="bold",
    edge_color="gray",
    width=1.5
)
plt.title("Friend Network Graph", fontsize=16)
plt.show()

# Step 4: Generate features and labels for machine learning
features = []
labels = []
nodes = list(G.nodes())
degree_centrality = nx.degree_centrality(G)

for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        features.append([
            degree_centrality[nodes[i]],  # Degree centrality of node i
            degree_centrality[nodes[j]],  # Degree centrality of node j
            len(list(nx.common_neighbors(G, nodes[i], nodes[j]))),  # Common neighbors
        ])
        labels.append(1 if G.has_edge(nodes[i], nodes[j]) else 0)

print("Class Distribution in Labels:")
print(Counter(labels))

# Step 5: Resample to balance classes
X = np.array(features)
y = np.array(labels)
X_majority = X[y == 0]
y_majority = y[y == 0]
X_minority = X[y == 1]
y_minority = y[y == 1]

X_minority_upsampled, y_minority_upsampled = resample(
    X_minority, y_minority,
    replace=True,
    n_samples=len(X_majority),
    random_state=42
)

X_balanced = np.vstack((X_majority, X_minority_upsampled))
y_balanced = np.hstack((y_majority, y_minority_upsampled))

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Step 7: Train the classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 8: Predict and calculate probabilities
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)[:, 1]

# Step 9: Calculate metrics
conf_matrix = confusion_matrix(y_test, predictions)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
report = classification_report(y_test, predictions)

# Print metrics
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {roc_auc:.2f}")

# Step 10: Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Step 1: Define a function to recommend friends
def recommend_friends(graph, person, top_n=5):
    if person not in graph.nodes:
        print(f"Person '{person}' not found in the network.")
        return []
    
    # Get the set of current friends
    current_friends = set(graph.neighbors(person))
    
    # Find potential friends (not already friends, excluding self)
    potential_friends = {}
    for node in graph.nodes:
        if node != person and node not in current_friends:
            # Count mutual friends (common neighbors)
            mutual_friends = len(list(nx.common_neighbors(graph, person, node)))
            if mutual_friends > 0:
                potential_friends[node] = mutual_friends
    
    # Sort potential friends by the number of mutual friends (descending)
    sorted_recommendations = sorted(potential_friends.items(), key=lambda x: x[1], reverse=True)
    
    # Return the top N recommendations
    return sorted_recommendations[:top_n]

# Step 1: Get the first 51 people from the graph or dataset
people_to_recommend = list(G.nodes)[:51]  # Select the first 51 nodes from the graph

# Step 2: Generate recommendations for each person
all_recommendations = {}

for person in people_to_recommend:
    recommendations = recommend_friends(G, person)
    all_recommendations[person] = recommendations


file = open('friend_recommendations.txt','w')
# Step 3: Print recommendations for each person
for person, recommendations in all_recommendations.items():
    file.write(f"Top friend recommendations for {person}:\n") # Write to file, add newline
    for friend, mutual_count in recommendations:
        file.write(f"  {friend} (Mutual Friends: {mutual_count})\n") # Write to file, add newline
    file.write("\n")  # Add an extra newline for better readability in the file

file.close()