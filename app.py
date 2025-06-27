# Importing necessary libraries for the flash application and the main project logic
from flask import Flask, render_template, request
from finalprojectREMASTERED import main

# Initialize the Flask application
app = Flask(__name__)

# Defining the route for the index page, accessible via GET request
@app.route("/", methods=["GET"])
def index():
    # Render the index.html template with no initial message and success set to True
    return render_template("index.html", message=None, success=True)

# Define the route for form submission, accessible via POST request
@app.route("/submit", methods=["POST"])
def submit():
    # Begin try block to handle potential errors during form processing
    try:
        # Retrieve and clean the username from the form data
        userName = request.form.get("userName", "").strip()
        stocks = []
        for i in range(1, 6):                        # get stock information from stock1 to stock5
            # Get stock name, remove leading '$' and clean input
            stock_name = request.form.get(f"stockName{i}", "").strip().lstrip('$')
            # Get holding days
            holding_days = request.form.get(f"holdingDays{i}", "").strip()
            # Get stock quantity
            stock_quantity = request.form.get(f"stockQuantity{i}", "").strip()
            # Validate that all stock fields are filled
            if not stock_name or not holding_days or not stock_quantity:
                return render_template("index.html", message="Error: All stock codes, days, and quantities must be filled in.", success=False)
            # Ensure holding days and stock quantity are both integer
            try:
                holding_days = int(holding_days)
                stock_quantity = int(stock_quantity)
            except ValueError:
                return render_template("index.html", message="Error: Holding days and stock quantities must be integers.", success=False)
            # Append valid stock data to the stock
            stocks.append({"stockName": stock_name, "holdingDays": holding_days, "stockQuantity": stock_quantity})

        to_email = request.form.get("to_email", "").strip()
        sender_email = request.form.get("sender_email", "").strip()
        sender_password = request.form.get("sender_password", "").strip()

        # Validate that all required fields (username, emails, password) are filled
        if not all([userName, to_email, sender_email, sender_password]):
            return render_template("index.html", message="Error: All fields must be filled in.", success=False)

        # Call the main function with form data and capture success status and message
        success, message = main(userName, stocks, to_email, sender_email, sender_password)
        # Render the index.html template with the returned message and success status
        return render_template("index.html", message=message, success=success)
    except Exception as e:
        return render_template("index.html", message=f"Submission failed: {e}", success=False)

# Check if the script is run directly (not imported as a module)
if __name__ == "__main__":
    # Start the Flask development server in debug mode on localhost at port 5001 with a specified host id
    app.run(debug=True, host="127.0.0.1", port=5001)