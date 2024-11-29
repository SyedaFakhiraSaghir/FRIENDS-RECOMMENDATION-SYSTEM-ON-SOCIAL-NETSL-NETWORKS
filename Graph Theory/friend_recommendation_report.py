# Step 1: Read the recommendations file
with open('friend_recommendations.txt', 'r') as file:
    lines = file.readlines()

# Step 2: Parse the recommendations into a structured format
recommendations = {}
current_person = None

for line in lines:
    line = line.strip()
    if line.startswith("Top friend recommendations for"):
        current_person = line.split("for")[1].strip().replace(":", "")
        recommendations[current_person] = []
    elif "(" in line and current_person:
        friend, mutual_count = line.split("(")
        friend = friend.strip()
        mutual_count = int(mutual_count.split(":")[1].replace(")", ""))
        recommendations[current_person].append((friend, mutual_count))

# Step 3: Analyze the data
# 3.1 Total people analyzed
total_people = len(recommendations)

# 3.2 Number of people with at least one recommendation
people_with_recommendations = sum(1 for recs in recommendations.values() if recs)

# 3.3 Total and average recommendations
total_recommendations = sum(len(recs) for recs in recommendations.values())
avg_recommendations_per_person = total_recommendations / total_people if total_people > 0 else 0

# 3.4 Top recommended friends (by mutual count frequency)
all_recommended_friends = Counter(friend for recs in recommendations.values() for friend, _ in recs)
most_frequent_recommendations = all_recommended_friends.most_common(10)

# Step 4: Generate report
report = f"""
Friend Recommendation Report
============================
1. Total People Analyzed: {total_people}
2. People with Recommendations: {people_with_recommendations} ({(people_with_recommendations / total_people) * 100:.2f}%)
3. Total Recommendations Made: {total_recommendations}
4. Average Recommendations Per Person: {avg_recommendations_per_person:.2f}
5. Most Frequently Recommended Friends:
"""

for rank, (friend, count) in enumerate(most_frequent_recommendations, 1):
    report += f"   {rank}. {friend} - Recommended {count} times\n"

# Write the report to a file
with open('recommendation_report.txt', 'w') as report_file:
    report_file.write(report)

print("Report generated: 'recommendation_report.txt'")
