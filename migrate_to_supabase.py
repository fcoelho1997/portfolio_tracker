"""
Run this once to upload your existing portfolio.csv trades to Supabase.
Usage: python migrate_to_supabase.py
"""
import pandas as pd
from supabase import create_client

url = input("Paste your Supabase Project URL: ").strip()
key = input("Paste your Supabase anon/public key: ").strip()
sb  = create_client(url, key)

df = pd.read_csv("portfolio.csv", parse_dates=["date"])
count = 0
for _, row in df.iterrows():
    sb.table("trades").insert({
        "ticker":     row["ticker"],
        "date":       str(row["date"].date()),
        "quantity":   float(row["quantity"]),
        "price_paid": float(row["price_paid"]),
    }).execute()
    count += 1
    print(f"  Uploaded {row['ticker']} {row['date'].date()}")

print(f"\nDone — {count} trades migrated to Supabase.")
