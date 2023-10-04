import sqlite3

# Ouvrir une connexion à la base de données SQLite
conn = sqlite3.connect('database.db')

# Créer une table 'users' avec une colonne pour l'ID, le nom d'utilisateur et le mot de passe
conn.execute('''CREATE TABLE users
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL);''')

# Fermer la connexion à la base de données SQLite
conn.close()