from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os

app = Flask(__name__, static_folder=".", static_url_path="")

RESULTS_DIR = "results"

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


def load_quantum_csv(hour):
    path = os.path.join(RESULTS_DIR, f"quantum_allocation_hour_{hour}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Quantum result file not found")
    return pd.read_csv(path)


@app.route("/load_results", methods=["POST"])
def load_results():
    hour = int(request.json["hour"])
    df = load_quantum_csv(hour)
    return jsonify({
        "quantum": df.to_dict(orient="records")
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    hour = int(data["hour"])
    question = data["question"].lower()

    df = load_quantum_csv(hour)

    total_demand = df["predicted_demand_mw"].sum()
    total_allocated = df["allocated_mw"].sum()
    remaining = total_demand - total_allocated

    # كم ضل كهربا
    if any(k in question for k in ["remaining", "left", "ضل"]):
        return jsonify({
            "answer": f"Allocated electricity is {total_allocated:.2f} MW. Remaining unmet demand is {remaining:.2f} MW."
        })

    # مين أخذ كهربا
    if any(k in question for k in ["who", "regions", "areas"]):
        lines = []
        for _, r in df.iterrows():
            lines.append(
                f"{r['region']} – {r['facility_type']}: "
                f"{int(r['allocation_level']*100)}% ({r['allocated_mw']} MW)"
            )
        return jsonify({"answer": "\n".join(lines)})

    # ليش التوزيع
    if "why" in question:
        return jsonify({
            "answer": (
                "Allocation is based on priority level, outage risk, predicted demand, "
                "and limited supply. Critical facilities are served first, then partial allocation is applied fairly."
            )
        })

    return jsonify({
        "answer": "Ask me about remaining electricity, allocated regions, or why some areas received partial supply."
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)

