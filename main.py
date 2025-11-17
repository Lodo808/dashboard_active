from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import datetime
import bcrypt
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re


app = Flask(__name__)

db_user = os.environ["DB_USER"]
db_pass = os.environ["DB_PASS"]
db_name = os.environ["DB_NAME"]
JWT_SECRET = os.environ["JWT_SECRET"]
instance_connection_name = os.environ["INSTANCE_CONNECTION_NAME"]

# 🔌 Connessione tramite socket Cloud SQL
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql+pymysql://{db_user}:{db_pass}@/"
    f"{db_name}?unix_socket=/cloudsql/{instance_connection_name}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------ MODELLO UTENTI ------------------
class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    operator = db.Column(db.String(100), nullable=False)
    company_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


# ------------------ LOGIN ------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
        return jsonify({"error": "Invalid credentials"}), 401

    # 🔥 CREA JWT
    payload = {
        "username": user.username,
        "company_name": user.company_name,
        "operator": user.operator,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30)
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    return jsonify({
        "status": "Login successful",
        "username": user.username,
        "company_name": user.company_name,
        "operator": user.operator,
        "token": token
    }), 200

@app.route("/testdb", methods=["GET"])
def test_db():
    try:
        db.session.execute(db.text("SELECT 1"))
        return jsonify({"message": "✅ Connessione al database riuscita!"}), 200
    except Exception as e:
        return jsonify({"error": f"❌ Errore nella connessione al database: {str(e)}"}), 500


# ------------------ AGGIUNTA DATI ------------------
@app.route("/add_reading", methods=["POST"])
def add_reading():
    data = request.get_json()
    username = data.get("username")

    # Timestamp MySQL safe
    try:
        datetime.datetime.fromisoformat(data.get("timestamp", datetime.datetime.now().isoformat()))
    except Exception:
        data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 🔍 Recupera utente + azienda
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    company = re.sub(r'[^a-z0-9]+', '_', user.company_name.lower())
    table_name = f"readings_{company}"

    # 🔧 Crea tabella aziendale se non esiste
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            operator_name VARCHAR(50),
            device_name VARCHAR(50),
            timestamp DATETIME,
            bluetooth_value DOUBLE PRECISION,
            effective_temperature DOUBLE PRECISION,
            qr_code VARCHAR(100),
            desired_temperature VARCHAR(20),
            item_id VARCHAR(50),
            label_datetime VARCHAR(50),
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            province VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    db.session.execute(text(create_table_sql))
    db.session.commit()

    fields = [
        "operator_name", "device_name", "timestamp", "bluetooth_value",
        "effective_temperature", "qr_code", "desired_temperature",
        "item_id", "label_datetime", "latitude", "longitude", "province"
    ]

    values = {}
    for f in fields:
        val = data.get(f)
        values[f] = val if val not in [None, "", " "] else "Unknown"

    placeholders = ", ".join([f":{f}" for f in fields])
    insert_sql = text(f"INSERT INTO {table_name} ({', '.join(fields)}) VALUES ({placeholders})")
    db.session.execute(insert_sql, values)
    db.session.commit()

    return jsonify({
        "status": "✅ Data inserted",
        "company": user.company_name,
        "operator": user.operator
    }), 200


# ------------------ GET READINGS ------------------
@app.route("/get_readings", methods=["POST"])
def get_readings():
    data = request.get_json()
    username = data.get("username")
    qr_code = data.get("qr_code")

    if not username:
        return jsonify({"error": "Missing username"}), 400

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    company = re.sub(r'[^a-z0-9]+', '_', user.company_name.lower())
    table_name = f"readings_{company}"

    try:
        query = f"SELECT * FROM {table_name}"
        params = {}

        if qr_code:
            query += " WHERE qr_code = :qr_code"
            params["qr_code"] = qr_code

        query += " ORDER BY timestamp DESC"

        result = db.session.execute(text(query), params)
        readings = [dict(row._mapping) for row in result]

        if not readings:
            return jsonify([]), 200  # NON errore, lista vuota

        # Convert datetime to string
        for r in readings:
            for k, v in r.items():
                if isinstance(v, datetime.datetime):
                    r[k] = v.isoformat()

        return jsonify(readings), 200

    except Exception as e:
        if "doesn't exist" in str(e):
            return jsonify([]), 200  # tabella vuota = nessun errore
        return jsonify({"error": str(e)}), 500


# ------------------ MAIN ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
