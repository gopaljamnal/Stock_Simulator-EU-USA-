# app.py
from flask import Flask, request, jsonify
# from flask_cors import CORS
import numpy as np
import json
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes


# European Option Pricing using Monte Carlo with Geometric Brownian Motion
@app.route('/api/european-option', methods=['POST'])
def european_option():
    # Parse parameters from request
    params = request.json
    S0 = float(params.get('S0', 100))  # Initial stock price
    K = float(params.get('K', 100))  # Strike price
    T = float(params.get('T', 1))  # Time to maturity (years)
    r = float(params.get('r', 0.05))  # Risk-free rate
    sigma = float(params.get('sigma', 0.2))  # Volatility
    simulations = int(params.get('simulations', 1000))  # Number of simulations
    steps = int(params.get('steps', 252))  # Time steps (trading days in a year)

    # Calculate time step size
    dt = T / steps

    # Initialize arrays for simulation
    np.random.seed(42)  # For reproducibility
    Z = np.random.normal(0, 1, (simulations, steps))

    # Initialize stock price array (simulations x steps+1)
    S = np.zeros((simulations, steps + 1))
    S[:, 0] = S0  # Set initial stock price

    # Simulate stock price paths using Geometric Brownian Motion
    for t in range(1, steps + 1):
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    # Calculate option payoff at maturity (for call option)
    payoffs = np.maximum(S[:, -1] - K, 0)

    # Option price is the expected payoff discounted to present value
    option_price = np.exp(-r * T) * np.mean(payoffs)

    # Select sample paths for visualization (first 10 paths)
    sample_paths = []
    for i in range(min(10, simulations)):
        path = [{"t": t * dt, "price": float(S[i, t])} for t in range(steps + 1)]
        sample_paths.append(path)

    # Prepare response
    result = {
        "optionPrice": float(option_price),
        "payoffMean": float(np.mean(payoffs)),
        "discount": float(np.exp(-r * T)),
        "paths": sample_paths
    }

    return jsonify(result)


# American Option Pricing using Least Squares Monte Carlo
@app.route('/api/american-option', methods=['POST'])
def american_option():
    # Parse parameters from request
    params = request.json
    S0 = float(params.get('S0', 100))  # Initial stock price
    K = float(params.get('K', 100))  # Strike price
    T = float(params.get('T', 1))  # Time to maturity (years)
    r = float(params.get('r', 0.05))  # Risk-free rate
    sigma = float(params.get('sigma', 0.2))  # Volatility
    simulations = int(params.get('simulations', 1000))  # Number of simulations
    steps = int(params.get('steps', 50))  # Time steps

    # Calculate time step size
    dt = T / steps

    # Discount factor for one time step
    discount = np.exp(-r * dt)

    # Initialize stock price array
    np.random.seed(42)  # For reproducibility
    S = np.zeros((simulations, steps + 1))
    S[:, 0] = S0

    # Simulate stock price paths
    for t in range(1, steps + 1):
        Z = np.random.standard_normal(simulations)
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    # Initialize cash flow matrix for put option
    cash_flows = np.zeros((simulations, steps + 1))

    # Terminal payoff for American put
    cash_flows[:, -1] = np.maximum(K - S[:, -1], 0)

    # Early exercise boundary
    exercise_boundary = np.zeros(steps + 1)
    exercise_boundary[-1] = K  # At expiration, exercise if S < K

    # Backward induction for optimal exercise
    for t in range(steps - 1, 0, -1):
        # Identify in-the-money paths
        itm = K > S[:, t]

        if np.sum(itm) > 0:
            # Stock prices at time t for ITM options
            S_itm = S[itm, t]

            # Cash flows at t+1 discounted to t
            Y = cash_flows[itm, t + 1] * discount

            # Regression basis functions (1, S, S^2)
            X = np.column_stack((np.ones(len(S_itm)), S_itm, S_itm ** 2))

            # Least squares regression (X'X)^-1 X'Y
            try:
                beta = np.linalg.lstsq(X, Y, rcond=None)[0]

                # Expected continuation value
                C = X.dot(beta)

                # Immediate exercise value
                exercise = np.maximum(K - S_itm, 0)

                # Exercise when immediate value > continuation value
                exercise_idx = exercise > C

                if np.any(exercise_idx):
                    # Update cash flows
                    itm_indices = np.where(itm)[0]
                    exercise_now = itm_indices[exercise_idx]

                    cash_flows[exercise_now, t] = exercise[exercise_idx]
                    cash_flows[exercise_now, t + 1:] = 0  # No future cash flows if exercised

                    # Update exercise boundary (average stock price where exercise is optimal)
                    if np.sum(exercise_idx) > 0:
                        exercise_boundary[t] = np.mean(S_itm[exercise_idx])
            except np.linalg.LinAlgError:
                # Handle singular matrix case
                pass

    # Calculate option price (present value of cash flows)
    option_values = np.zeros(simulations)
    for i in range(simulations):
        # Find the first non-zero cash flow (time of exercise)
        exercise_times = np.where(cash_flows[i, :] > 0)[0]
        if len(exercise_times) > 0:
            t_exercise = exercise_times[0]
            option_values[i] = cash_flows[i, t_exercise] * np.exp(-r * t_exercise * dt)

    option_price = np.mean(option_values)

    # Select sample paths for visualization
    sample_paths = []
    for i in range(min(10, simulations)):
        path = [{"t": t * dt, "price": float(S[i, t])} for t in range(steps + 1)]
        sample_paths.append(path)

    # Format exercise boundary for chart
    boundary_data = [{"t": t * dt, "price": float(exercise_boundary[t])}
                     for t in range(steps + 1) if exercise_boundary[t] > 0]

    # Prepare response
    result = {
        "optionPrice": float(option_price),
        "paths": sample_paths,
        "exerciseBoundary": boundary_data
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
