from flask import Flask, render_template, request
import pickle

app = Flask(__name__, template_folder='C:/Users/umai/Desktop/projects/classifier1')

# Load the trained model
with open("model.pkl", "rb") as file:
    clf = pickle.load(file)

# Load the fitted vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the review text from the textarea
        review = request.form["review"]
        orig_review = request.form["review"]
        # Convert the review text into numerical data
        review = vectorizer.transform([review])
        
        # Predict the sentiment of the review
        sentiment = clf.predict(review)[0]
        
        # Return the sentiment to the template
        return render_template("/index.html", sentiment=sentiment, orig_review=orig_review)
    return render_template("/index.html")

if __name__ == "__main__":
    app.run(debug=True)
