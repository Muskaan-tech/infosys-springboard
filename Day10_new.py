from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Global counters
fake_count = 0
real_count = 0
last_prediction = None


@app.route('/')
def home():
    return render_template('index1_updated.html',
                           fake=fake_count,
                           real=real_count,
                           last_prediction=last_prediction)


@app.route('/predict', methods=['POST'])
def predict():
    global fake_count, real_count, last_prediction

    job_desc = request.form.get('job_description', '').strip()

    # ✅ Error Handling
    if not job_desc:
        return render_template('index1_updated.html',
                               error="Please enter a job description.",
                               fake=fake_count, real=real_count,
                               last_prediction=last_prediction)

    if not re.search('[a-zA-Z]', job_desc):
        return render_template('index1_updated.html',
                               error="Job description must contain letters, not only numbers/symbols.",
                               fake=fake_count, real=real_count,
                               last_prediction=last_prediction)

    if len(job_desc.split()) < 5:
        return render_template('index1_updated.html',
                               error="Please enter at least 5 words for better prediction.",
                               fake=fake_count, real=real_count,
                               last_prediction=last_prediction)

    # ✅ Prediction
    X_input = vectorizer.transform([job_desc])
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][1]

    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob * 100, 2) if pred == 1 else round((1 - prob) * 100, 2)
    color = "fake" if pred == 1 else "real"

    # Update counters
    if pred == 1:
        fake_count += 1
    else:
        real_count += 1

    # Save last prediction
    last_prediction = {"label": label, "confidence": confidence, "color": color}

    return render_template('result1_updated.html',
                           label=label,
                           confidence=confidence,
                           color=color,
                           description=job_desc,
                           fake=fake_count,
                           real=real_count)


if __name__ == '__main__':
    app.run(debug=True)
