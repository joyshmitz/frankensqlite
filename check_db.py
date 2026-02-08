import sqlite3
try:
    conn = sqlite3.connect('spec_evolution_v1.sqlite3')
    c = conn.cursor()
    c.execute('SELECT max(add_lines) FROM commits')
    print(f"Max added lines: {c.fetchone()[0]}")
    c.execute('SELECT count(*) FROM commits')
    print(f"Total commits: {c.fetchone()[0]}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")
