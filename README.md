This project is a simple AI-powered chatbot built using Python, TensorFlow/Keras, NLTK, and Flask. It uses an intents.json file to understand user 
questions and generate responses based on a trained neural network model. The train.py script tokenizes patterns using NLTK, prepares training data, 
and saves the trained model as model.h5 along with words.pkl and classes.pkl. The Flask app (app.py) loads the model and provides a web interface located in 
the templates and static folders, allowing users to interact with the chatbot in real time. To run the project, install dependencies from requirements.txt,
download NLTK packages (punkt and punkt_tab), train the model if needed, and start the Flask server using python app.py to access the chatbot at 
http://127.0.0.1:5000. The project is fully customizable—adding new patterns and responses to intents.json and retraining the model will allow you to expand 
the chatbot’s knowledge and behavior.
